# pdLSR: Pandas-aware least squares regression

`pdLSR` is a Pandas-aware least squares minimization library.

## Overview

`pdLSR` is a library for performing least squares minimization. It attempts to seamlessly incorporate this task in a Pandas-focused workflow. Input data are expected in dataframes, and multiple regressions can be performed using functionality similar to Pandas `groupby`. Results are returned as grouped dataframes and include best-fit parameters, statistics, residuals, and more. The results can be easily visualized using [`seaborn`](https://github.com/mwaskom/seaborn).

`pdLSR` currently utilizes [`lmfit`](https://github.com/lmfit/lmfit-py), a flexible and powerful library for least squares minimization, which in turn, makes use of `scipy.optimize.leastsq`. I began using `lmfit` because it is one of the few libraries that supports non-linear least squares regression, which is commonly used in the natural sciences. I also like the flexibility it offers for testing different modeling scenarios and the variety of assessment statistics it provides. However, I found myself writing many `for` loops to perform regressions on groups of data and aggregate the resulting output. Simplification of this task was my inspiration for writing `pdLSR`.

`pdLSR` is related to libraries such as [`statsmodels`](http://statsmodels.sourceforge.net) and [`scikit-learn`](http://scikit-learn.org/stable/) that provide linear regression functions that operate on dataframes. However, these libraries don't support grouping operations on dataframes and don't aggregate output into dataframes. Supporting `statsmodels` and `scikit-learn` is being considered. (And pull requests adding this functionality would be welcome.)

Some additional 'niceties' associated with the input of parameters and equations have also been incorporated. `pdLSR` also utilizes multithreading for the calculation of confidence intervals, as this process is time consuming when there are more than a few groups.

## Setup

### Dependencies

The following libraries are required for `pdLSR`:  

* numpy
* pandas
* lmfit
* multiprocess  

[`multiprocess`](https://github.com/uqfoundation/multiprocess) is a fork of Python's `multiprocessing` library that provides more robust multithreading. I found that this library is required for multithreading to work with `pdLSR`. Both `multiprocess` and `lmfit` will install automatically from `pip` or `conda` (see below).

For plotting, `matplotlib` is required and `seaborn` is recommended. 

`pdLSR` works with Python 2 and 3.

### Installation and Demo
[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/mlgill/pdLSR)

The preferred method for installing `pdLSR` and all of its dependencies is to use the `conda` or `pip` package managers. 

* For conda: `conda install -c mlgill pdlsr` -- unfortunately conda seems to require lowercase names for packages
* For pip: `pip install pdLSR`

However it can also be installed manually by cloning the repo into your `PYTHONPATH`.  

There is a demo notebook that can be executed locally or live from GitHub using [mybinder.org](http://mybinder.org/repo/mlgill/pdLSR). After clicking the badge at the top of this section, navigate to `pdLSR --> demo --> pdLSR_demo.ipynb` and everything should be setup to execute the demo in a browser. No installation required!

## Documentation

The functions of `pdLSR` are documented within the code, but currently the best single source for using `pdLSR` is the [demo notebook](https://github.com/mlgill/pdLSR/blob/master/pdLSR/demo/pdLSR_demo.ipynb). Developing stand-alone documentation is a future goal.


