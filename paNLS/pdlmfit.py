import pandas as pd
import numpy as np
from six import string_types

from .fitting import get_fits, convert_param_dict_to_df, convert_param_df_to_expanded_list, get_confidence_interval, predict
from .aggregation import get_results, get_data, get_stats


class pdlmfit(object):
    
    def __init__(self, data, func=None, groupcols=None, params=None, 
                 xname=None, yname=None, yerr=None, 
                 method='leastsq', sigma=0.95, threads=None):

        if groupcols is not None:
            if ( (not hasattr(groupcols, '__iter__')) | isinstance(groupcols, string_types) ):
                groupcols = [groupcols]

        self._input_data = data
        self._func = func
        self._groupcols = groupcols
        self._params = params
        self._xname = xname
        self._yname = yname
        self._yerr = yerr
        self._method = method
        self._sigma = sigma
        self._threads = threads

        self._index = ( data[groupcols + [xname]]
                        .groupby(groupcols)
                        .max()
                        .index
                       )
        self._ngroups = self._index.shape[0]

        return
    

    def fit(self):

        # Expand the parameters if necessary
        params = self._params
        if isinstance(params[0], dict):
            params = convert_param_dict_to_df(params, self._ngroups, self._index)

        params = convert_param_df_to_expanded_list(params)
        paramnames = [[x.name for x in row] for row in params]

        
        # Perform the regressions and aggregate results
        self._fit_data = get_fits(self._input_data, self._func, self._groupcols, params,
                                  self._xname, self._yname, self._yerr, 
                                  self._method, self._sigma, self._threads)


        # Parameter name manipulation
        self._fit_data['paramnames'] = paramnames
        self._paramnames = list(np.unique(np.array([x.name for x in row for row in paramnames])))


        # Calculate dof to determine if error estimation is possible
        self._fit_data['nobs'] = self._fit_data.fitobj.apply(lambda x: x.ndata)
        self._fit_data['npar'] = self._fit_data.params.apply(lambda x: len([y.name for y in x if y.vary]))
        self._fit_data['dof'] = self._fit_data.nobs - self._fit_data.npar

        # Perform the confidence interval calculation, but ensure only data with >= 2 parameters is used
        self._fit_data['ciobj'] = np.NaN
        if (self._fit_data.npar.max() > 1):
            bool_mask = (self._fit_data.npar > 1)
            self._fit_data.ix[bool_mask, 'ciobj'] = get_confidence_interval(self._fit_data.ix[bool_mask], 
                                                                            self._sigma, self._threads)

        self.results = get_results(self._fit_data, self._paramnames)
        self.data = get_data(self._fit_data, self._paramnames, self._groupcols)
        self.stats = get_stats(self._fit_data, self.data)
       
        return


    def predict(self, xtype='global', xnum=50, xcalc=None):

        # Perform a prediction
        self.model = predict(self._fit_data, self._groupcols,
                             xtype=xtype, xnum=xnum, xcalc=xcalc)

        return

    # def ftest(self,fitobj2):
    #     chisq_table = ftest(self,fitobj2,self.groupcols)
        
    #     self.fstatistic = chisq_table
    #     # fitobj2.ftest = chisq_table

        return