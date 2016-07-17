DOCSTRING = """\
pdLSR
by Michelle L. Gill

`pdLSR` is a library for performing least squares regression. It attempts to 
seamlessly incorporate this task in a Pandas-focused workflow. Input data 
are expected in dataframes, and multiple regressions can be performed using 
functionality similar to Pandas `groupby`. Results are returned as grouped 
dataframes and include best-fit parameters, statistics, residuals, and more. 

`pdLSR` has been tested on python 2.7, 3.4, and 3.5. It requires Numpy, 
Pandas, multiprocess (https://github.com/uqfoundation/multiprocess), and 
lmfit (https://github.com/lmfit/lmfit-py). All dependencies are installable
via pip or conda (see README.md).

A demonstration notebook is provided in the `demo` directory or the demo
can be run via GitHub (see README.md).

"""
