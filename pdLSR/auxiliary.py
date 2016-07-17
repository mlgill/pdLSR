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


# # Function to expand arrays contained in a single dataframe row
# def expand_df(data, name, groupby):
    
#     if len(groupby) == 1:
#         groupby = groupby[0]
    
#     if isinstance(data, pd.DataFrame):
#         dfexpand = pd.concat([pd.Series(np.array(x[1].values[0]), 
#                                         index=pd.Index([x[0]]*len(x[1].values[0]), name=groupby),
#                                         name=name) 
#                               for x in data.iterrows()], axis=0)
#     else:
#         dfexpand = pd.concat([pd.Series(np.array(x[1]),
#                                         index=pd.Index([x[0]]*len(x[1]), name=groupby),
#                                         name=name) 
#                               for x in data.iteritems()], axis=0)
        
#     if not(isinstance(dfexpand, pd.DataFrame)):
#         dfexpand = pd.DataFrame(dfexpand)
        
#     return dfexpand
    