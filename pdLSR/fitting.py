import lmfit
import pandas as pd
import numpy as np
from .auxiliary import error_function

import multiprocess as multiprocessing


def get_minimizer(index, fitobj_df, data_df, params_df, 
                  xname, yname, yerr):
    """Create an indexed series of minimizer functions.

    Parameters
    ----------
    index : Pandas index
        A grouped index for the dataframe.
    fitobj_df : dataframe
        A dataframe of the fit objects.
    data_df : dataframe
        A dataframe of the input data.
    params_df : dataframe
        A dataframe of the input parameters.
    xname : string
        Name of the data_df column containing the xdata.
    yname : string
        Name of the data_df column contain the ydata.
    yerr : string
        Name of the data_df column containg the yerror
        estimates (optional).


    Returns
    -------
    minimizer : series
        A pandas series containing all the indexed minimizer objects."""
    
    # Creates the lmfit minimizer object
    
    minimizer = list()

    for i in index:

        model_eq = fitobj_df.loc[i, 'model_eq']

        xdata = data_df.loc[i, xname].values
        ydata = data_df.loc[i, yname].values

        if yerr is not None:
            yerrors = data_df.loc[i, yerr].values
            fcn_args = '(model_eq, xdata, ydata, yerrors)'
        else:
            fcn_args = '(model_eq, xdata, ydata)'


        params = params_df.loc[i]
        params = [ lmfit.Parameter(**params.loc[x].to_dict()) 
                   for x in params.index.levels[0]
                 ]

        minimizer.append(lmfit.Minimizer(error_function, 
                                         params, 
                                         fcn_args=eval(fcn_args))
                        )

    return pd.Series(minimizer, index=index, name='minimizer')


def get_confidence_interval(fitobj_df, mask, sigma, threads=None):
    """Calculate the confidence intervals from the minimizer objects.

    Parameters
    ----------
    fitobj_df : dataframe
        A dataframe of the fit objects.
    mask : array/series
        A boolean mask indicating if the confidence intervals
        can't be calculated for any of the groups.
    sigma : float or list
        The confidence intervals to be calculated.
    threads : int or None
        The number of threads to use for multithreaded 
        calculation of confidence intervals.

    Returns
    -------
    ci_obj : series
        A series of confidence interval objects."""

    def _calc_ci(arr):

        ciobj = list()
        for m,f in zip(arr[:,0], arr[:,1]):
            try:
                res = lmfit.conf_interval(m, f, sigmas=sigma)
            except:
                res = np.NaN

            ciobj.append(res)

        return np.array(ciobj)

    if threads is None:
        threads = multiprocessing.cpu_count()

    fitobj_arr = np.array_split(fitobj_df[['minimizer','fitobj']].values, threads)

    pool = multiprocessing.Pool()
    ciobj = np.array(pool.map(_calc_ci, fitobj_arr)).flatten()
    
    return pd.Series(ciobj, index=fitobj_df.index)
