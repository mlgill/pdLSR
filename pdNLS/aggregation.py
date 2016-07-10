import pandas as pd
import re


# Create the results table
def get_results(fitobj_df, params, sigma):
       
    # Get the parameters
    parameters = pd.concat( [fitobj_df.fitobj.apply(lambda x: pd.Series({'{}_value'.format(par_name):x.params[par_name].value, 
                                                                         '{}_stderr'.format(par_name):x.params[par_name].stderr}
                                                                     )
                                                )
                            for par_name in params ], axis=1)
    
    # Get the confidence interval bound for each % ci entered
    mask = pd.notnull(fitobj_df.ciobj)
    conf_intervals = pd.concat([fitobj_df.loc[mask, 'ciobj'].apply(lambda x: pd.Series({'{}_ci{:.2f}'.format(par_name, x[par_name][s][0]):
                                                                                        x[par_name][s][1]}))
                                for par_name in params 
                                for s in range(len(sigma))
                               ], axis=1)

    # Combine results and fix column names
    results = pd.merge(parameters, conf_intervals, left_index=True, right_index=True)
    
    colnames = [re.search(r"""([^_]+)_([^_]+)""", col) for col in results.columns]
    results.columns = pd.MultiIndex.from_tuples([(col.group(1), col.group(2)) for col in colnames])
    
    results.sortlevel(axis=1, inplace=True)
    
    # Fix the confidence intervals so they are differences
    # This requires slicing the data and dropping the last column
    # index to get the broadcasing working
    ci_cols = [x for x in results.columns.levels[-1] if 'ci' in x]

    value_df = results.loc[:, (slice(None),['value'])]
    value_df.columns = value_df.columns.droplevel(-1)

    for ci in ci_cols:
        ci_df = results.loc[:, (slice(None),[ci])]
        ci_cols_orig = ci_df.columns
        ci_df.columns = ci_df.columns.droplevel(-1)

        ci_df = (value_df - ci_df).abs()
        ci_df.columns = ci_cols_orig

        results.loc[:, ci_df.columns] = ci_df
    
    results = (results
               .sortlevel(level=1, axis=1, ascending=False)
               .sortlevel(level=0, axis=1, sort_remaining=False)
               )
    
    return results


# The stats table
def get_stats(fitobj_df, stats_df, stats_cols=['chisqr', 'redchi', 'aic', 'bic', 'covar']):
    
    for dat in stats_cols:
        lambda_str = 'lambda x: x.{}'.format(dat)
        stats_df[dat] = fitobj_df.apply(eval(lambda_str))
        
    return stats_df


# Table of the xdata, ydata, ycalc, and residuals
# def get_data(fit_data, paramnames, groupcols):

#     # Index must be reset to avoid shape error with > 1 groupcol
#     ycalc = fit_data.reset_index().apply(lambda x: tuple( x.model_eq([x.fitobj.params[par].value for par in paramnames], 
#                                                           np.asarray(x.xdata)) 
#                                                          ), axis=1)

#     # ycalc.set_index(groupcols, inplace=True)
#     ycalc = fix_index(ycalc, fit_data, groupcols, 'ycalc')

#     # The residuals from lmfit are wrong, so calculate them below
#     # resid = fit_data.fitobj.apply(lambda x: tuple( x.residual ))

#     data = pd.concat([ expand_df(fit_data.xdata, 'xdata', groupcols),
#                        expand_df(fit_data.ydata, 'ydata', groupcols),
#                        expand_df(ycalc, 'ycalc', groupcols)
#                      ], axis=1)

#     data['residual'] = data.ydata - data.ycalc
    
#     return data