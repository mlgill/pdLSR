# pdNLS: Pandas-aware non-linear least squares minimization

`pdNLS` is a Pandas-aware non-linear least squares minimization library that uses [`lmfit`](https://github.com/lmfit/lmfit-py/) for regression.

## Overview

`pdNLS` is a library for performing non-linear least squares (NLS) minimization. It attempts to seamlessly incorporate this task in a Pandas-focused workflow. Input data are expected in dataframes, and multiple regressions can be performed using functionality similar to Pandas `groupby`. Results are returned as grouped dataframes and include best-fit parameters, statistics, residuals, and more. The results can be easily visualized using [`seaborn`](https://github.com/mwaskom/seaborn).

`pdNLS` is related to libraries such as [`statsmodels`](http://statsmodels.sourceforge.net) and [`scikit-learn`](http://scikit-learn.org/stable/) that provide linear regression functions that operate on dataframes. As I was unable to find any that perform non-linear regression, I developed `pdNLS` to fill this niche. 

`pdNLS` utilizes [`lmfit`](https://github.com/lmfit/lmfit-py), a flexible and powerful library for non-linear least squares minimization, which in turn, makes use of `scipy.optimize.leastsq`. I began using `lmfit` several years ago because I like the flexibility it offers for testing different modeling scenarios and the variety of assessment statistics it provides. However, I found myself writing many `for` loops to perform regressions on groups of data and aggregate the resulting output.

Some additional 'niceties' associated with the input of parameters and equations have also been incorporated. `pdNLS` also utilizes multithreading for the calculation of confidence intervals, as this process is time consuming when there are more than a few groups.

## Setup

### Dependencies

The following libraries are required for `pdNLS`:  

* numpy
* pandas
* lmfit
* multiprocess  

[`multiprocess`](https://github.com/uqfoundation/multiprocess) is a fork of Python's `multiprocessing` library that provides more robust multithreading. I found that this library is required for multithreading to work with `pdNLS`. Both `multiprocess` and `lmfit` will install automatically from `pip` or `conda` (see below).

For plotting, the following additional libraries are required:  

* matplotlib
* seaborn

`pdNLS` has currently only been tested with Python 2, but this is expected to change in the near future.

### Installation and Demo
[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/mlgill/pdNLS)

The preferred method for installing `pdNLS` and all of its dependencies is to use the `conda` or `pip` package managers. 

* For conda: `conda install -c mlgill pdnls` -- unfortunately conda seems to require lowercase names for packages
* For pip: `pip install pdNLS`

However it can also be installed manually by cloning the repo into your `PYTHONPATH`.  

There is a demo notebook that can be executed locally or live from GitHub using [mybinder.org](http://mybinder.org/repo/mlgill/pdNLS). After clicking the badge at the top of this section, navigate to `pdNLS --> demo --> pdNLS_demo.ipynb` and everything should be setup to execute the demo in a browser. No installation required!

## Documentation

The functions of `pdNLS` are documented within the code, but currently the best single source for using `pdNLS` is the [demo notebook](https://github.com/mlgill/pdNLS/blob/master/pdNLS/demo/pdNLS_demo.ipynb). Developing stand-alone documentation is a future goal.


