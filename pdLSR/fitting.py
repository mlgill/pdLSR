import lmfit
import pandas as pd
import numpy as np
from .auxiliary import error_function

import multiprocess as multiprocessing


def get_minimizer(index, fitobj_df, data_df, params_df, 
                  xname, yname, yerr):
    
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


# # Function to predict y-values based on fit parameters and an x-array
# def predict(fit_data, groupcols, xtype='global', xnum=50, xcalc=None):
    
#     model = pd.DataFrame(index=fit_data.index)
#     model['model_eq'] = fit_data.model_eq

#     model['params'] = fit_data.reset_index().apply(lambda x: tuple([x.fitobj.params[par].value 
#                                                                     for par in x.paramnames]), axis=1).tolist()

#     if xcalc is not None:
#         model['xcalc'] = [tuple(xcalc) for x in range(model.shape[0])]
#     else:
#         if xtype == 'global':
#             xmin = fit_data.xdata.apply(lambda x: np.array(x).min())
#             xmax = fit_data.xdata.apply(lambda x: np.array(x).max())
#             model['xcalc'] = [tuple(np.linspace(xmin.min(), xmax.max(), xnum)) for x in range(model.shape[0])]
#         else:
#             model['xcalc'] = fit_data.reset_index().apply(lambda x: tuple(np.linspace(np.asarray(x.xdata).min(),
#                                                                                       np.asarray(x.xdata).max(),
#                                                                                       xnum)),
#                                                           axis=1).tolist()

#     model['ycalc'] = model.reset_index().apply(lambda x: 
#                                            tuple( x.model_eq(x.params, np.asarray(x.xcalc)) ), axis=1).tolist()

#     predict = pd.concat([ expand_df(model.xcalc, 'xcalc', groupcols), 
#                           expand_df(model.ycalc, 'ycalc', groupcols) ], axis=1)

#     return predict