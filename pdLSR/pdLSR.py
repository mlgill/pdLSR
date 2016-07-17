import pandas as pd
import numpy as np
from six import string_types

from .fitting import get_minimizer, get_confidence_interval
from .aggregation import get_results, get_stats, get_covar
from .lmfit_setup import lmfit_params


class pdLSR(object):

    """pdLSR is a class that performs least-squares regression using
    Pandas-aware features such as dataframe input and groupby functionality.

    Parameters
    ----------
    data : dataframe
        Input data containing dependent, independent, and optional 
        error estimates.
    model_eq : function 
        The function that will be minimized during regression.
    groupby : list or string
        The name of a column (or a list of columns) to use for grouping
        during regression.
    xname : string
        The column name of the independent variable. Currently only 
        a single independent variable is supported.
    yname : string
        The column name of the dependent variable. Currently only
        a single dependent variable is supported.
    yerr : string or None
        The column name of the error estimates (optional). Currently
        only a single column of error estimates is supported.
    minimizer : string
        The name of the minimizer function. 'lmfit' is currently
        supported.
    minimizer_kwargs : dictionary
        The arguments for the minimizer. 

        For 'lmfit':
        Required inputs are:
            'params': the parameters. 
        Optional inputs are:
            'method': only 'leastsq' is supported.
            'sigma': which is the confidence interval 
                (number or list, default is 0.95).
            'threads': the number of threads to use for calculation of the 
                confidence intervals (default is None which will cause
                automatic calculation).

    Returns
    -------
    pdLSR : class
        The pdLSR class contains the `fit` and `predict` methods to
        perform minimization and calculate a best fit line based on
        input values.
    """
    
    def __init__(self, data, model_eq, groupby, 
                 xname, yname, yerr=None, 
                 minimizer='lmfit',
                 **minimizer_kwargs):

        """Initialize the pdLSR class and check that inputs are valid."""

        # Ensure the selected columns aren't hidden in the index
        data = data.reset_index()

        # Setup the groupby columns
        if ( (not hasattr(groupby, '__iter__')) | isinstance(groupby, string_types) ):
            groupby = [groupby]

        for col in groupby:
            if col not in data.columns:
                raise IndexError('{} not in input data columns or index.'.format(col))
                
        self._groupby = groupby
        self._ngroupby = len(groupby)

        # Dependent and independent variables
        self._xname = xname
        self._yname = yname
        self._yerr = yerr

        self._datacols = [self._xname, self._yname]

        if self._yerr is not None:
            self._datacols += [self._yerr]

        # Check that all columns are in the data
        for col in self._datacols:
            if col not in data.columns:
                raise IndexError('{} not in input data columns or index.'.format(col))

        # Unique index information
        index = ( data[self._groupby + [self._xname]]
                  .groupby(self._groupby)
                  .max()
                  .index
                 )

        self._index = index
        self._ngroups = data.index.shape[0]

        # Get the arguments for the minimizer
        kwargs_input = minimizer_kwargs['minimizer_kwargs']

        if minimizer=='lmfit':
            self = lmfit_params(self, kwargs_input)
        else:
            raise NotImplementedError('The only minimizer currently implemented is lmfit.')
        
        # Append the dataframe of data
        self.data = ( data
                     [self._groupby + self._datacols]
                     .set_index(self._groupby)
                     )

        # Dataframe to hold the fitting objects
        self._fitobj = pd.DataFrame(index=self._index)
        self._fitobj['model_eq'] = model_eq
        self._fitobj['model_eq'] = self._fitobj.model_eq.fillna(method='ffill')

        return

    
    def _predict(self, xcalc, xnum):

        """Perform prediction on input independent data or on a custom range."""
        
        xname = self._xname
        
        if xcalc:
            if xcalc=='global':
                xmin = self.data[xname].min()
                xmax = self.data[xname].max()
                xdata = np.linspace(xmin, xmax, xnum)
        
        predict_list = list()

        index_names = self._fitobj.index.names
        
        for index in self._fitobj.index.values:
            
            if xcalc:
                if xcalc=='local':
                    xmin = self.data.loc[index, xname].min()
                    xmax = self.data.loc[index, xname].max()
                    xdata = np.linspace(xmin, xmax, xnum)
            else:
                xdata = (self
                         .data
                         .loc[index, xname]
                         .squeeze()
                         .values
                         )

            model_eq = (self
                        ._fitobj
                        .loc[index, 'model_eq']
                        )
            
            params = (self
                      .results
                      .sortlevel(axis=1)
                      .loc[index, (slice(None), ['value'])]
                      .squeeze()
                      .values
                      )
            
            ydata = model_eq(params, xdata)
            
            index_array = pd.Index([index]*len(xdata), name=index_names)

            if xcalc:
                predict_data = pd.DataFrame({'xcalc':xdata, 'ycalc':ydata},
                                            index=index_array)
            else:
                predict_data = pd.Series(ydata, name='ycalc',
                                         index=index_array)
                
            predict_list.append(predict_data)
            
        return pd.concat(predict_list, axis=0)
    
    
    def fit(self):
        """Perform regression using the minimizer and input parameters of choice.

        Returns
        -------
        pdLSR.data : dataframe
            This dataframe contains the input data, predicted values, and
            residuals.
        pdLSR.results : dataframe
            This dataframe contains the regression parameters.
        pdLSR.stats : dataframe
            This dataframe contains the evaluation parameters provided by 
            the minimizer.
        pdLSR.covar : dataframe
            This dataframe contains the covariance matrix.
        """
        
        # Perform the minimization
        self._fitobj['minimizer'] = get_minimizer(self._index, self._fitobj, self.data, self._params_df, 
                                                  self._xname, self._yname, self._yerr)
        
        self._fitobj['fitobj'] = ( self._fitobj.minimizer
                                  .apply(lambda x: x.minimize(method=self._method))
                                  )

        self.data = self.data.sortlevel(axis=0)
        
        # Create the stats dataframe
        self.stats = pd.DataFrame(index=self._index)
        
        # dof calculations for confidence intervals
        self.stats['nobs'] = ( self
                              .data
                              .groupby(level=list(range(self._ngroupby)))
                              .size()
                              )

        self.stats['npar'] = len(self._params_df.columns.levels[0])
        self.stats['dof'] = self.stats.nobs - self.stats.npar
        
        # Get the confidence intervals
        mask = self.stats.npar > 1
        if mask.sum() > 0:
            self._fitobj['ciobj'] = get_confidence_interval(self._fitobj, 
                                                            mask, 
                                                            self._sigma, 
                                                            self._threads)
        else:
            self._fitobj['ciobj'] = np.NaN
            
        
        # The results
        self.results = ( get_results(self._fitobj, 
                                     self._paramnames, 
                                     self._sigma)
                         .sortlevel(axis=0)
                        )

        # Predict the y values and calculate residuals
        self.data['ycalc'] = self._predict(None, None)
        self.data['residuals'] = self.data[self._yname] - self.data['ycalc']
        
        # The remaining statistics
        self.stats = ( get_stats(self._fitobj.fitobj, 
                                 self.stats)
                       .sortlevel(axis=0)
                      )

        # The covariance table
        self.covar = get_covar(self._fitobj)
        
        return


    def predict(self, xcalc='global', xnum=20):
        """Calculate expected values based on interpolated independent data.

        Parameters
        ----------
        xcalc : string
            'global' or 'local' indicates whether to use the global min/max
            for dependent data or just those from the current group.
            'global' is the default.
        xnum : int
            Number of points to determine, default is 20.

        Returns
        -------
        pdLSR.model : dataframe
            An indexed dataframe with calculated independent and 
            dependent data.
        """

        if not hasattr(self, 'results'):
            raise AttributeError('The pdLSR object must be fit before predicting.')

        self.model = self._predict(xcalc, xnum)
        return


    def pivot_covar(self):
        """Pivot the covariance matrix so column values extend horizontally.

        Parameters
        ----------
        pdLSR : class
            A pdLSR class with covariance dataframe.

        Returns
        -------
        covar_pivot : dataframe
            A dataframe pivoted so that column elements of the matrix
            are horizonal. Useful for multiplication, etc."""

        if not hasattr(self, 'covar'):
            raise AttributeError('The pdLSR object does not contain a covariance matrix dataframe.')

        return (self
                .covar
                .groupby(level=list(range(self._ngroupby)))
                .apply(lambda x: x.pivot(index='row',
                                         columns='col',
                                         values='covar'))
                )
