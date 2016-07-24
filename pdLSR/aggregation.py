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

    # Extract the confidence intervals as a dictionary
    conf_intervals = pd.concat([fitobj_df.loc[mask, 'ciobj'].apply(lambda x: pd.Series({(par_name, val[1][0], val[0]):val[1][1] 
                                                                                        for val in enumerate(x[par_name])}))
                                for par_name in params], axis=1)

    # Drop the 0.0 sigma value which is just the parameter value itself
    conf_intervals = conf_intervals.loc[:, (slice(None), sigma)]

    conf_intervals.sort_index(axis=1, inplace=True)

    # Rename the integer level in preparation for joining to sigma value
    max_val = conf_intervals.columns.get_level_values(-1).max()
    mid_val = int((max_val+1)/2)
    column_mapper = dict([(x,'lo') for x in range(mid_val)] + 
                         [(x,'hi') for x in range(mid_val+1, max_val+1)])
    conf_intervals = conf_intervals.rename_axis(column_mapper, axis=1)

    # Set new indices
    level_2 = ['ci{}_{}'.format(x,y) for x,y in 
               zip(conf_intervals.columns.get_level_values(1), 
                   conf_intervals.columns.get_level_values(2))]

    ci_index = pd.MultiIndex.from_tuples(zip(conf_intervals.columns.get_level_values(0), level_2))

    conf_intervals.columns = ci_index
    
    # Fix column names for parameters
    colnames = [re.search(r"""([^_]+)_([^_]+)""", col) for col in parameters.columns]
    parameters.columns = pd.MultiIndex.from_tuples([(col.group(1), col.group(2)) for col in colnames])

    # Combine parameters and confidence intervals
    results = pd.merge(parameters, conf_intervals, left_index=True, right_index=True)
    
    results.sortlevel(axis=1, inplace=True)
    
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

    stats_df = stats_df.rename(columns={'redchi':'r_chisqr'})
        
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


