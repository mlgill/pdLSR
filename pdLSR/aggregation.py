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
def get_stats(fitobj_df, stats_df, stats_cols=['chisqr', 'redchi', 'aic', 'bic']):
    
    for dat in stats_cols:
        lambda_str = 'lambda x: x.{}'.format(dat)
        stats_df[dat] = fitobj_df.apply(eval(lambda_str))
        
    return stats_df


# The covariance table
def get_covar(fitobj_df):

    predict_list = list()

    index_names = fitobj_df.index.names
    
    covar_list = list()
    
    for index in fitobj_df.index.values:        
        covar = fitobj_df.loc[index, 'fitobj'].covar
        
        index_array = pd.Index([index]*4, name=index_names)
        
        covar_list.append(pd.DataFrame({'row':  [0,0,1,1],
                                        'col':  [0,1,0,1],
                                        'covar':covar.flatten()},
                                       index=index_array,
                                       columns=['row', 'col', 'covar']))
        
    return pd.concat(covar_list, axis=0)


