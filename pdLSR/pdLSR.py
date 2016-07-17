import pandas as pd
import numpy as np
from six import string_types

from .fitting import get_minimizer, get_confidence_interval
from .aggregation import get_results, get_stats, get_covar
from .auxiliary import convert_param_dict_to_df


class pdLSR(object):
    
    def __init__(self, data, model_eq, groupcols, params, 
                 xname, yname, yerr=None, 
                 method='leastsq', sigma=0.95, threads=None):
        
        # Ensure the selected columns aren't in the index
        data = data.reset_index()

        # Setup the groupby columns
        # TODO check that groupcols are in the data, otherwise quit
        if ( (not hasattr(groupcols, '__iter__')) | isinstance(groupcols, string_types) ):
            groupcols = [groupcols]
                
        self._groupcols = groupcols
        self._ngroupcols = len(groupcols)
        
        self._paramnames = [x['name'] for x in params]

        # Dependent and independent variables
        # TODO check that xname/yname are in the data, otherwise quit
        self._xname = xname
        self._yname = yname
        self._yerr = yerr

        self._datacols = [xname, yname]
        if yerr is not None:
            # TODO check that yerr is in the data, otherwise quit
            self._datacols += [yerr]

        # Append the dataframe of data
        self.data = ( data
                     .reset_index()
                     [self._groupcols + self._datacols]
                     .set_index(self._groupcols)
                     )

        # Unique index information
        index = ( data[self._groupcols + [xname]]
                  .groupby(self._groupcols)
                  .max()
                  .index
                 )

        self._index = index
        self._ngroups = data.index.shape[0]
        
        # Fitting and confidence interval methods
        self._method = method
        # TODO check that method is leastsq, otherwise quit
        
        if not hasattr(sigma, '__iter__'):
            sigma = [sigma]
        self._sigma = sigma
        
        self._threads = threads
        
        # Dataframe to hold the fitting objects
        self._fitobj = pd.DataFrame(index=self._index)
        self._fitobj['model_eq'] = model_eq
        self._fitobj['model_eq'] = self._fitobj.model_eq.fillna(method='ffill')
        
        # Setup parameter dataframe
        self._params = params
        if (isinstance(self._params, list) and isinstance(self._params[0], dict)):
            # params is a list of dictionaries
            params_df = convert_param_dict_to_df(self._params, 
                                                 self._ngroups, 
                                                 self._index)
            
        elif isinstance(self._params, pd.DataFrame):
            # params is a dataframe
            params_df = self._params
            
        else:
            # TODO make unrecognized parameter format throw an error and quit
            print('Parameters should be either a list of dictionaries or a dataframe.')
            
        self._params_df = params_df.fillna(method='ffill')

        return

    
    def _predict(self, xcalc, xnum):
        
        xname = self._xname
        
        if xcalc:
            if xcalc=='global':
                xmin = self.data[xname].min()
                xmax = self.data[xname].max()
                xdata = np.linspace(xmin, xmax, xnum)
        
        predict_list = list()

        index_names = self._fitobj.index.names
        
        for index in self._fitobj.index.values:
            
            if xcalc:
                if xcalc=='local':
                    xmin = self.data.loc[index, xname].min()
                    xmax = self.data.loc[index, xname].max()
                    xdata = np.linspace(xmin, xmax, xnum)
            else:
                xdata = (self
                         .data
                         .loc[index, xname]
                         .squeeze()
                         .values
                         )

            model_eq = (self
                        ._fitobj
                        .loc[index, 'model_eq']
                        )
            
            params = (self
                      .results
                      .sortlevel(axis=1)
                      .loc[index, (slice(None), ['value'])]
                      .squeeze()
                      .values
                      )
            
            ydata = model_eq(params, xdata)
            
            index_array = pd.Index([index]*len(xdata), name=index_names)

            if xcalc:
                predict_data = pd.DataFrame({'xcalc':xdata, 'ycalc':ydata},
                                            index=index_array)
            else:
                predict_data = pd.Series(ydata, name='ycalc',
                                         index=index_array)
                
            predict_list.append(predict_data)
            
        return pd.concat(predict_list, axis=0)
    
    
    def fit(self):
        
        # Perform the minimization
        self._fitobj['minimizer'] = get_minimizer(self._index, self._fitobj, self.data, self._params_df, 
                                                  self._xname, self._yname, self._yerr)
        
        self._fitobj['fitobj'] = ( self._fitobj.minimizer
                                  .apply(lambda x: x.minimize(method=self._method))
                                  )

        self.data = self.data.sortlevel(axis=0)
        
        # Create the stats dataframe
        self.stats = pd.DataFrame(index=self._index)
        
        # dof calculations for confidence intervals
        self.stats['nobs'] = ( self
                              .data
                              .groupby(level=range(self._ngroupcols))
                              .size()
                              )
        self.stats['npar'] = len(self._params_df.columns.levels[0])
        self.stats['dof'] = self.stats.nobs - self.stats.npar
        
        # Get the confidence intervals
        mask = self.stats.npar > 1
        if mask.sum() > 0:
            self._fitobj['ciobj'] = get_confidence_interval(self._fitobj, 
                                                            mask, 
                                                            self._sigma, 
                                                            self._threads)
        else:
            self._fitobj['ciobj'] = np.NaN
            
        
        # The results
        self.results = ( get_results(self._fitobj, 
                                     self._paramnames, 
                                     self._sigma)
                         .sortlevel(axis=0)
                        )

        # Predict the y values and calculate residuals
        self.data['ycalc'] = self._predict(None, None)
        self.data['residuals'] = self.data[self._yname] - self.data['ycalc']
        
        # The remaining statistics
        self.stats = ( get_stats(self._fitobj.fitobj, 
                                 self.stats)
                       .sortlevel(axis=0)
                      )

        # The covariance table
        self.covar = get_covar(self._fitobj)
        
        return


    def predict(self, xcalc='global', xnum=10):
        self.model = self._predict(xcalc, xnum)
        return


    def pivot_covar(self):
        return (self
                .covar
                .groupby(level=range(self._ngroupcols))
                .apply(lambda x: x.pivot(index='row',
                                         columns='col',
                                         values='covar'))
                )
