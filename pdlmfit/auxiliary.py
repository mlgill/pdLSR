import pandas as pd
import numpy as np


# Function to expand arrays contained in a single dataframe row
def expand_df(data, name, groupcols):
    
    if len(groupcols) == 1:
        groupcols = groupcols[0]
    
    if isinstance(data, pd.DataFrame):
        dfexpand = pd.concat([pd.Series(np.array(x[1].values[0]), 
                                        index=pd.Index([x[0]]*len(x[1].values[0]), name=groupcols),
                                        name=name) 
                              for x in data.iterrows()], axis=0)
    else:
        dfexpand = pd.concat([pd.Series(np.array(x[1]),
                                        index=pd.Index([x[0]]*len(x[1]), name=groupcols),
                                        name=name) 
                              for x in data.iteritems()], axis=0)
        
    if not(isinstance(dfexpand, pd.DataFrame)):
        dfexpand = pd.DataFrame(dfexpand)
        
    return dfexpand


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


# Function to fix indexing for multiple groupcols
def fix_index(data, fit_data, groupcols, name):

    data.name = name
    data = pd.DataFrame(data)

    idx_data = fit_data.reset_index()[groupcols]

    for col in groupcols:
        data[col] = idx_data[col]

    data.set_index(groupcols, inplace=True)

    return data
    