import pandas as pd
import numpy as np
from six import string_types

from .fitting import get_minimizer, get_confidence_interval
from .aggregation import get_results, get_stats
from .auxiliary import convert_param_dict_to_df


class pdNLS(object):
    
    def __init__(self, data, model_eq=None, groupcols=None, params=None, 
                 xname=None, yname=None, yerr=None, 
                 method='leastsq', sigma=0.95, threads=None):
        
        # Ensure the selected columns aren't in the index
        data = data.reset_index()

        # Setup the groupby columns
        # TODO check that groupcols are in the data, otherwise quit
        if groupcols is not None:
            if ( (not hasattr(groupcols, '__iter__')) | isinstance(groupcols, string_types) ):
                groupcols = [groupcols]
                
        self._groupcols = groupcols
        self._ngroupcols = len(groupcols)
        
        self._paramnames = [x['name'] for x in params]

        # Dependent and independent variables
        # TODO check that xname/yname/yerr are in the data, otherwise quit
        self._xname = xname
        self._yname = yname
        self._yerr = yerr

        self._datacols = [xname, yname]
        if yerr is not None:
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
        # TODO search for NaNs in model column and fill
        
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
    
    def _predict(self, data_df, xcalc, xtype, xnum):
    
        # Get the index for selection
        index = data_df.index.unique()[0]

        # Choose the xdata
        print(self)
        xname = self._xname
        xdata = self.data.loc[index, xname]

        if xcalc:
            if xtype == 'global':
                xmin = self.data[xname].min()
                xmax = self.data[xname].max()
            else:
                xmin = xdata.min()
                xmax = xdata.max()

            xname = 'xdata'
            xdata = pd.Series(np.linspace(xmin, xmax, xnum))
            xindex = pd.Index(np.array([index]*xnum))

        # Pick out the parameters for this data
        params = ( self
                 .results
                 .sortlevel(axis=1)
                 .loc[[index], (slice(None),['value'])]
                 )
        params.columns = params.columns.levels[0]
        params = params[self._paramnames].squeeze()


        model_eq = self._fitobj.model_eq.loc[index]

        ycalc = model_eq(params, xdata)

        if not xcalc:
            return ycalc
        else:
            return pd.DataFrame({xname:xdata, 'ycalc':ycalc})
    
    
    def predict(self, xcalc=False, xtype='global', xnum=50):
        
        retval = ( self
                  ._fitobj
                  .groupby(level=range(self._ngroupcols), 
                            group_keys=False)
                  .apply(lambda x: _predict(x, xcalc, xtype, xnum))
                  )
                  
        return retval
    
    
    def fit(self):
        
        # Perform the minimization
        self._fitobj['minimizer'] = get_minimizer(self._index, self._fitobj, self.data, self._params_df, 
                                                  self._xname, self._yname, self._yerr)
        
        self._fitobj['fitobj'] = ( self._fitobj.minimizer
                                  .apply(lambda x: x.minimize(method=self._method))
                                  )
        
        # TODO make the predict function work
        # Predict the y values and calculate residuals
        #self.data['ycalc'] = ( self
        #                      .data
        #                      .groupby(level=range(self._ngroupcols), 
        #                               group_keys=False)
        #                      .apply(lambda x: predict(x))
        #                      )
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
        
        # The remaining statistics
        self.stats = ( get_stats(self._fitobj.fitobj, 
                                 self.stats)
                       .sortlevel(axis=0)
                      )
        
        return