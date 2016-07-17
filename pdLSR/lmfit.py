import pandas as pd

def lmfit_params(self, kwargs_input):

    kwargs = {'method':'leastsq',
              'sigma':'0.95',
              'threads':None}

    kwargs.update(kwargs_input)

    if 'params' in kwargs.keys():
        self._params = kwargs['params']
    else:
        print('params are a required input')
        # TODO: quit with an error


    # Fitting and confidence interval methods
    self._method = kwargs['method']
    # TODO check that method is leastsq, otherwise quit

    self._sigma = kwargs['sigma']

    if not hasattr(self._sigma, '__iter__'):
        self._sigma = [self._sigma]

    self._threads = kwargs['threads']

    # Setup parameter dataframe
    self._paramnames = [x['name'] for x in self._params]

    if (isinstance(self._params, list) and isinstance(self._params[0], dict)):
        # params is a list of dictionaries
        params_df = convert_param_dict_to_df(self._params, 
                                             self._ngroups, 
                                             self._index)
        
    elif isinstance(self._params, pd.DataFrame):
        # params is a dataframe
        params_df = self._params
        
    else:
        # TODO make unrecognized parameter format throw an error and quit
        print('Parameters should be either a list of dictionaries or a dataframe.')
        
    self._params_df = params_df.fillna(method='ffill')

    return self


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