# pdNLS: Pandas-aware non-linear least squares minimization

`pdNLS` is a Pandas-aware non-linear least squares minimization library that uses [`lmfit`](https://github.com/lmfit/lmfit-py/) for regression.

## Overview

`pdNLS` is a library for performing non-linear least squares (NLS) minimization. It attempts to seamlessly incorporate this task in a Pandas-focused workflow. Input data are expected in dataframe format, and multiple regressions can be performed using functionality similar to Pandas `groupby`. Results are returned as grouped dataframes and include best-fit parameters, statistics, residuals, and more. The results can be easily visualized using [`seaborn`](https://github.com/mwaskom/seaborn).

`pdNLS` is related to libraries such as [`statsmodels`](http://statsmodels.sourceforge.net) and [`scikit-learn`](http://scikit-learn.org/stable/) that provide linear regression functions that operate on dataframes. However, I was unable to find any that perform non-linear regression. I developed `pdNLS` to fill this niche. 

`pdNLS` utilizes [`lmfit`](https://github.com/lmfit/lmfit-py), a flexible and powerful library for non-linear least squares minimization, which in turn, makes use of `scipy.optimize.leastsq`. I began using `lmfit` several years ago because I like the flexibility it offers for testing different modeling scenarios and the variety of assessment statistics it provides. However, I found myself writing many `for` loops to perform regressions on groups of data and aggregate the resulting output.

Some additional 'niceties' associated with the input of parameters and equations have also been incorporated. `pdNLS` also utilizes multithreading for the calculation of confidence intervals, as this process is time consuming when there are more than a few groups.

## Binder
[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/mlgill/pdNLS)
