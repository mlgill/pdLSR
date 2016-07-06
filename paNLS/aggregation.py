import pandas as pd
import numpy as np

from .auxiliary import expand_df, fix_index


# Create the results table
def get_results(fit_data, paramnames):

    param_values = fit_data.fitobj.apply(lambda x: [x.params[par].value for par in paramnames])
    param_stderr = fit_data.fitobj.apply(lambda x: [x.params[par].stderr for par in paramnames])

    if 'ciobj' in fit_data.columns:
        bool_mask = pd.notnull(fit_data.ciobj)
        param_cilo   =  fit_data.ix[bool_mask, 'ciobj'].apply(lambda x: [x[par][0][1] for par in paramnames])
        param_cihi   =  fit_data.ix[bool_mask, 'ciobj'].apply(lambda x: [x[par][-1][1] for par in paramnames])

    results = pd.DataFrame(index=fit_data.index)

    for par in enumerate(paramnames):
        results[par[1]] = param_values.apply(lambda x: x[par[0]])

    for par in enumerate(paramnames):
        results[par[1]+'_SE'] = param_stderr.apply(lambda x: x[par[0]])

    if 'ciobj' in fit_data.columns:
        for par in enumerate(paramnames):
            results[par[1]+'_LO'] = np.NaN
            results[par[1]+'_HI'] = np.NaN
            
            results.ix[bool_mask, par[1]+'_LO'] = param_cilo.apply(lambda x: x[par[0]])
            results.ix[bool_mask, par[1]+'_HI'] = param_cilo.apply(lambda x: x[par[0]])
        
    return results



# Table of the xdata, ydata, ycalc, and residuals
def get_data(fit_data, paramnames, groupcols):

    # Index must be reset to avoid shape error with > 1 groupcol
    ycalc = fit_data.reset_index().apply(lambda x: tuple( x.model_eq([x.fitobj.params[par].value for par in paramnames], 
                                                          np.asarray(x.xdata)) 
                                                         ), axis=1)

    # ycalc.set_index(groupcols, inplace=True)
    ycalc = fix_index(ycalc, fit_data, groupcols, 'ycalc')

    # The residuals from lmfit are wrong, so calculate them below
    # resid = fit_data.fitobj.apply(lambda x: tuple( x.residual ))

    data = pd.concat([ expand_df(fit_data.xdata, 'xdata', groupcols),
                       expand_df(fit_data.ydata, 'ydata', groupcols),
                       expand_df(ycalc, 'ycalc', groupcols)
                     ], axis=1)

    data['residual'] = data.ydata - data.ycalc
    
    return data



# The stats table
def get_stats(fit_data, output_data):
    stats = pd.DataFrame(index=fit_data.index)

    #stats['nobs'] = fit_data.fitobj.apply(lambda x: x.ndata)
    #stats['npar'] = fit_data.params.apply(lambda x: len([y.name for y in x if y.vary]))
    #stats['dof'] = stats.nobs - stats.npar
    stats['nobs'] = fit_data['nobs']
    stats['npar'] = fit_data['npar']
    stats['dof'] = fit_data['dof']

    stats['chisq'] = fit_data.fitobj.apply(lambda x: x.chisqr)
    stats['redchisq'] = fit_data.fitobj.apply(lambda x: x.redchi)
    stats['aic'] = fit_data.fitobj.apply(lambda x: x.aic)
    stats['bic'] = fit_data.fitobj.apply(lambda x: x.bic)
    stats['rss'] = output_data.residual.groupby(output_data.index).apply(lambda x: np.sum(x**2))

    # TODO: return this as a separate dataframe with an additional index
    stats['covar'] = [x[1].covar for x in fit_data.fitobj.iteritems()]
    
    return stats