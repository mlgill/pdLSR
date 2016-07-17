import pandas as pd
import numpy as np
import re


def get_results(fitobj_df, params, sigma):
    """Aggregate fit parameters into a dataframe.

    Parameters
    ----------
    fitobj_df : dataframe
        A dataframe containing minimizer objects and confidence intervals.
    params : list
        List of parameter names.
    sigma : float or list
        Confidence interval value or list.

    Returns
    -------
    results : dataframe
        A dataframe of parameter values and error estimates."""
       
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


def get_stats(fitobj_df, stats_df, stats_cols=['chisqr', 'redchi', 'aic', 'bic']):
    """Aggregate regression statistics into a dataframe.

    Parameters
    ----------
    fitobj_df : series
        A series containing minimizer objects.
    stats_df : dataframe
        A dataframe of previously calculated stats (dof, npar, etc.).
    stats_cols : list
        A list containing the names of fit attributes to add to the
        nascent stats dataframe.

    Returns
    -------
    stats_df : dataframe
        A dataframe with new"""
    
    for dat in stats_cols:
        lambda_str = 'lambda x: x.{}'.format(dat)
        stats_df[dat] = fitobj_df.apply(eval(lambda_str))
        
    return stats_df


def get_covar(fitobj_df):
    """Converte the covariance matrices into an indexed dataframe.

    Parameters
    ----------
    fitobj_df : dataframe
        A dataframe containing minimizer objects.

    Returns
    -------
    covar_df : dataframe
        A dataframe containing indexed covariance matrices."""

    predict_list = list()

    index_names = fitobj_df.index.names
    
    covar_list = list()
    
    # Iterate through each group, get the covariance matrix and put into 
    # a dataframe
    for index in fitobj_df.index.values:        
        covar = fitobj_df.loc[index, 'fitobj'].covar
        
        index_array = pd.Index([index]*4, name=index_names)

        # Handle the indexing of the matrix
        nrow,ncol = covar.shape
        row_index, col_index = np.unravel_index(list(range(nrow*ncol)), (nrow, ncol))
        
        covar_list.append(pd.DataFrame({'row':  row_index,
                                        'col':  col_index,
                                        'covar':covar.flatten()},
                                       index=index_array,
                                       columns=['row', 'col', 'covar']))
        
    return pd.concat(covar_list, axis=0)


