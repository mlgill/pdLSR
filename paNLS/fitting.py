import lmfit
import pandas as pd
import numpy as np
from .auxiliary import expand_df, error_function, fix_index

try:
    import multiprocess as multiprocessing
except:
    import multiprocessing


# Perform the regression
def get_fits(data, func, groupcols, params, 
             xname, yname, yerr=None, 
             method='leastsq', sigma=0.95, threads=None):

	# Create the dataframe
    #index = data[groupcols + [xname]].groupby(groupcols).min().index
    #fit_data = pd.DataFrame(index=index)
    fit_data = pd.DataFrame()


    # Add the xdata, ydata, and errors
    fit_data['xdata'] = data[groupcols + [xname]].groupby(groupcols).apply(lambda x: tuple(x[xname].values))
    fit_data['ydata'] = data[groupcols + [yname]].groupby(groupcols).apply(lambda x: tuple(x[yname].values))
    if yerr is not None:
        fit_data['yerr'] = data[groupcols + [yerr]].groupby(groupcols).apply(lambda x: tuple(x[yerr].values))
    
    
    # Add the equation and params--this works for a single function or a list of functions
    fit_data['model_eq'] = func
    fit_data['params'] = params



    # if yerr is None:
    #     fcn_args = "(np.asarray(x.xdata), np.asarray(x.ydata))"
    # else:
    #     fcn_args = "(np.asarray(x.xdata), np.asarray(x.ydata), np.asarray(x.yerr))"

    # fit_data['minimizer'] = fit_data.apply(lambda x: lmfit.Minimizer(x['model_eq'], x['params'],
    #                                                     fcn_args=eval(fcn_args)), axis=1)

    if yerr is None:
        fcn_args = "(x.model_eq, np.asarray(x.xdata), np.asarray(x.ydata))"
    else:
        fcn_args = "(x.model_eq, np.asarray(x.xdata), np.asarray(x.ydata), np.asarray(x.yerr))"

    fit_data['minimizer'] = fit_data.apply(lambda x: lmfit.Minimizer(error_function, x['params'],
                                                        fcn_args=eval(fcn_args)), axis=1)


    fit_data['fitobj'] = fit_data.minimizer.apply(lambda x: x.minimize(method=method))    

    return fit_data



def convert_param_dict_to_df(params, ngroups, index):
    ### CLEAN ###

    # Unique list of variable names
    var_names = map(lambda x: x['name'], params)

    # Expanded list of names and all properties to create a 
    # multi-level column index
    column_list = [(x['name'], y) 
                   for x in params 
                       for y in x.keys()
                  ]

    column_index = pd.MultiIndex.from_tuples(column_list, 
                                             sortorder=None)

    # Create a dataframe from the indexes and fill it
    param_df = pd.DataFrame(index=index, columns=column_index)

    # Fill df by iterating over the parameter name index
    # and then each of its keys
    for var in enumerate(param_df.columns.levels[0]):
        for key in param_df.loc[:,var[1]]:
            param_df[(var[1],key)] = params[var[0]][key]
            
    param_df.fillna(method='ffill', inplace=True)

    return param_df



def convert_param_df_to_expanded_list(param_df):
    ### CLEAN ###

    # Convert each parameter entry to a list of dictionaries
    list_of_dicts = [ [ param_df.loc[x,y].to_dict() 
                        for y in param_df.columns.levels[0] ] 
                      for x in param_df.index ]

    # Convert the list of dictionaries to a list of lmfit parameters
    param_list = [ [ lmfit.Parameter(**r) 
                     for r in row ] 
                  for row in list_of_dicts ]

    return param_list



# Return confidence interval
def get_confidence_interval(fit_data, sigma, threads=None):

    # For multithreaded implementation
    def _apply_df(args):
        df, func, kwargs = args
        return df.apply(eval(func), axis=1, **kwargs)

    def apply_by_multiprocessing(df, func, **kwargs):
        threads = kwargs.pop('threads')
        pool = multiprocessing.Pool(processes=threads)
        result = pool.map(_apply_df, [(d, func, kwargs)
                                      for d in np.array_split(df, threads)])
        pool.close()
        return pd.concat(list(result))

    # Make this a string and use eval in _apply_df to get around a pickling issue with lmfit.conf_interval/lambda
    # Alternatively, could try multiprocessing with dill as the pickler b/c it seems dill can pickle this
    #ci_func = 'lambda x: try: lmfit.conf_interval(x.minimizer, x.fitobj, sigmas=[{:.2f}]) except: np.NaN'.format(sigma)
    ci_func = 'lambda x: lmfit.conf_interval(x.minimizer, x.fitobj, sigmas=[{:.2f}])'.format(sigma)

    if threads is None:
        threads = multiprocessing.cpu_count()

    # Try to use parallelized version first, switch to unparallelized if error occurs
    # Only select portions of fit data that have two or more parameters

    fd_noidx = fit_data.copy().reset_index()
    try:
        ciobj = apply_by_multiprocessing(fd_noidx.ix[fd_noidx.npar>1], ci_func, threads=threads)
        print('try loop')
    except:
        ciobj = fd_noidx.ix[fd_noidx.npar>1].apply(lambda x: lmfit.conf_interval(x.minimizer, 
                                                             x.fitobj, sigmas=[sigma]), axis=1)
        print('except loop')

    return ciobj.tolist()


# def get_confidence_interval(fitobj_df, mask, sigma, threads=None):

#     if mask.sum() == 0:
#         print('There is not enough data to calculate a confidence interval.')
#         # TODO exit here

#     # For multithreaded implementation
#     def _apply_df(args):
#         df, func, kwargs = args
#         return df.apply(eval(func), axis=1, **kwargs)

#     def apply_by_multiprocessing(df, func, **kwargs):
#         threads = kwargs.pop('threads')
#         pool = multiprocessing.Pool(processes=threads)
#         result = pool.map(_apply_df, [(d, func, kwargs)
#                                       for d in np.array_split(df, threads)])
#         pool.close()
#         return pd.concat(list(result))

#     # Make this a string and use eval in _apply_df to get around a pickling issue with lmfit.conf_interval/lambda
#     # Right know this module also prefers multiprocessing with dill, which may still be more robust
#     ci_func = 'lambda x: lmfit.conf_interval(x.minimizer, x.fitobj, sigmas=[{:.2f}])'.format(sigma)

#     if threads is None:
#         threads = multiprocessing.cpu_count()

#     # This is a bit of a hack and is required to get this to work with dataframes
#     fitobj_noidx = fitobj_df.loc[mask, ['minimizer','fitobj']].reset_index(drop=True)

#     # Try to use parallelized version first, switch to unparallelized if error occurs 
#     try:
#         ciobj = apply_by_multiprocessing(fitobj_noidx, 
#                                          ci_func, 
#                                          threads=threads).dropna(axis=1).squeeze()
#     except:
#         print('Unable to use multithreading for confidence interval calculation. This might take a while.')
#         ciobj = fitobj_noidx.apply(lambda x: lmfit.conf_interval(x.minimizer, 
#                                                                  x.fitobj, 
#                                                                  sigmas=[sigma]), 
#                                           axis=1)

#     ciobj.set_axis(0, fitobj_df.loc[mask].index)
#     ciobj.name = 'ciobj'

#     return ciobj


# Function to predict y-values based on fit parameters and an x-array
def predict(fit_data, groupcols, xtype='global', xnum=50, xcalc=None):
    
    model = pd.DataFrame(index=fit_data.index)
    model['model_eq'] = fit_data.model_eq

    model['params'] = fit_data.reset_index().apply(lambda x: tuple([x.fitobj.params[par].value 
                                                                    for par in x.paramnames]), axis=1).tolist()

    if xcalc is not None:
        model['xcalc'] = [tuple(xcalc) for x in range(model.shape[0])]
    else:
        if xtype == 'global':
            xmin = fit_data.xdata.apply(lambda x: np.array(x).min())
            xmax = fit_data.xdata.apply(lambda x: np.array(x).max())
            model['xcalc'] = [tuple(np.linspace(xmin.min(), xmax.max(), xnum)) for x in range(model.shape[0])]
        else:
            model['xcalc'] = fit_data.reset_index().apply(lambda x: tuple(np.linspace(np.asarray(x.xdata).min(),
                                                                                      np.asarray(x.xdata).max(),
                                                                                      xnum)),
                                                          axis=1).tolist()

    model['ycalc'] = model.reset_index().apply(lambda x: 
                                           tuple( x.model_eq(x.params, np.asarray(x.xcalc)) ), axis=1).tolist()

    predict = pd.concat([ expand_df(model.xcalc, 'xcalc', groupcols), 
                          expand_df(model.ycalc, 'ycalc', groupcols) ], axis=1)

    return predict



# Function to return the parameters
def get_parameters(params, ngroups):
    
    # Optional keys for lmfit.Parameters
    lmfit_opt_parameter_keys = ['vary', 'min', 'max', 'expr', 'stderr', 'correl']
    
    output_parameters = list()

    # Iterate over each row of the groups
    for row in range(ngroups):

        single_row = list()

        # Iterate over each parameter
        for par in params:

            single_par = dict()

            # Each parameter requires a name and a value
            # The name must be identical, thus will be a single value
            single_par['name'] = par['name']

            # Value can be a single value or list-like
            if hasattr(par['value'], '__iter__'):
                single_par['value'] = np.asarray(par['value'])[row]
            else:
                single_par['value'] = par['value']

            # The remaining parameters are optional and can be list-like
            for opt in lmfit_opt_parameter_keys:

                if opt in par.keys():

                    # The correl option is a dictionary and has an __iter__ attribute,
                    # so this option won't work with 'np.asarray'
                    if (hasattr(par[opt], '__iter__') and (not isinstance(par[opt], dict))):
                        single_par[opt] = np.asarray(par[opt])[row]
                    else:
                        single_par[opt] = par[opt]


            # Create a parameter and append
            single_row.append(lmfit.Parameter(**single_par))

        # Append all parameters for a single row
        output_parameters.append(single_row)
        
    return output_parameters