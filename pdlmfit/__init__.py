from .pdlmfit import pdlmfit


# TODOs
# - [x] prediction function
# - [x] handle parameters on input
# - [x] handle multiple equations on input
# - [/] handle models with different parameter lengths?
# - [x] don't calculate confidence interval for single parameter regression
# - [x] fix calculation of nparam if parameter is fixed
# - [ ] paramnames need to be entered if there are different models?
# - [ ] check list lengths for params, models
# - [ ] linear regression module
# - [x] combine into single class
# - [ ] check if groupcols, xcol, xdata, yerr are in data
# - [ ] add kwargs to stuff for flexibility?
# - [ ] return warning if some fits fail, return success/failure messages
# - [ ] skip confidence interval if standard error is zero
# - [ ] test with different regression types--some don't return cov matrix, for example
# - [ ] incorporate lmfit model in my model method
