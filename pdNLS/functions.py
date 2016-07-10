import numpy as np

def linear(par, xdata):
    
    # Parse multiple input parameter
    # formats for slope, intercept    
    if hasattr(par,'valuesdict'):
        # lmfit parameter format
        var = par.valuesdict()
        slope = var['slope']
        intercept = var['intercept']
    elif hasattr(par,'keys'):
        # dict format
        slope = par['slope']
        intercept = par['intercept']
    else:
        # array/list/tuple format
        slope = par[0]
        intercept = par[1]

    # Calculate the y-data from the parameters
    return intercept + slope * xdata


def exponential_decay(par, xdata):
    
    # Parse multiple input parameter
    # formats for intensity, rate    
    if hasattr(par,'valuesdict'):
        # lmfit parameter format
        var = par.valuesdict()
        inten = var['inten']
        rate = var['rate']
    elif hasattr(par,'keys'):
        # dict format
        inten = par['inten']
        rate = par['rate']
    else:
        # array/list/tuple format
        inten = par[0]
        rate = par[1]

    # Calculate the y-data from the parameters
    return inten * np.exp(-1*rate*xdata)