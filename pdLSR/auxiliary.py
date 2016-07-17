import pandas as pd


# Loss function for error calculation
def error_function(par, func, xdata, ydata=None, yerr=None):
    
    # The calculated value
    ycalc = func(par, xdata)
    
    if ydata is None:
        # Calculation only
        return ycalc
    elif yerr is None:
        # Error minimization
        return (ycalc - ydata)**2
    else:
        # Error minimization with weights
        return (ycalc - ydata)**2/yerr**2


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
    params_df = pd.DataFrame(index=index, columns=column_index)

    # Fill df by iterating over the parameter name index
    # and then each of its keys
    for var in enumerate(params_df.columns.levels[0]):
        for key in params_df.loc[:,var[1]]:
            params_df[(var[1],key)] = params[var[0]][key]

    return params_df


# # Function to expand arrays contained in a single dataframe row
# def expand_df(data, name, groupcols):
    
#     if len(groupcols) == 1:
#         groupcols = groupcols[0]
    
#     if isinstance(data, pd.DataFrame):
#         dfexpand = pd.concat([pd.Series(np.array(x[1].values[0]), 
#                                         index=pd.Index([x[0]]*len(x[1].values[0]), name=groupcols),
#                                         name=name) 
#                               for x in data.iterrows()], axis=0)
#     else:
#         dfexpand = pd.concat([pd.Series(np.array(x[1]),
#                                         index=pd.Index([x[0]]*len(x[1]), name=groupcols),
#                                         name=name) 
#                               for x in data.iteritems()], axis=0)
        
#     if not(isinstance(dfexpand, pd.DataFrame)):
#         dfexpand = pd.DataFrame(dfexpand)
        
#     return dfexpand
    