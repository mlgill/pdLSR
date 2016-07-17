import pandas as pd

def lmfit_params(self, kwargs_input):
    """Assign lmfit minimizer parameters.

    Parameters
    ----------
    kwargs_input : dictionary
        Dictionary of input parameters for lmfit minimizer.

    Returns
    -------
    pdLSR : class
        Assigns necessary attributes to the pdLSR class."""

    kwargs = {'method':'leastsq',
              'sigma':'0.95',
              'threads':None}

    kwargs.update(kwargs_input)

    if 'params' in kwargs.keys():
        self._params = kwargs['params']
    else:
        raise AttributeError('"params" are a required input for the lmfit minimizer.')

    # Fitting and confidence interval methods
    self._method = kwargs['method']

    if self._method is not 'leastsq':
        raise NotImplementedError('The only lmfit minimization method currently implemented is leastsq.')

    self._sigma = kwargs['sigma']

    if not hasattr(self._sigma, '__iter__'):
        self._sigma = [self._sigma]

    self._threads = kwargs['threads']

    if (isinstance(self._params, list) and isinstance(self._params[0], dict)):
        # params is a list of dictionaries
        params_df = convert_param_dict_to_df(self._params, 
                                             self._ngroups, 
                                             self._index)
        
    elif isinstance(self._params, pd.DataFrame):
        # params is a dataframe
        params_df = self._params
        
    else:
        raise AttributeError('Parameters should be either a list of dictionaries or a dataframe.')
        
    self._params_df = params_df.fillna(method='ffill')

    # Setup parameter dataframe
    self._paramnames = [x['name'] for x in self._params]

    return self


def convert_param_dict_to_df(params, ngroups, index):
    """Converts a dictionary of parameter values to a dataframe.

    Parameters
    ----------
    params : dictionary
        Dictionary of parameter values for lmfit.
    ngroups : integer
        Number of groups being fit.
    index : Pandas index
        The index for the grouping of the dataframe.

    Returns
    -------
    params_df : dataframe
        An indexed dataframe with parameter values extracted."""

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