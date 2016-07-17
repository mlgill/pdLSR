from .functions import *
from .pdLSR import pdLSR

from .docstring import DOCSTRING
from .version import VERSION

__doc__ = DOCSTRING
__version__ = VERSION


# TODOs:
# [ ] add docstrings
# [ ] add .sortlevel(axis=0) to all tables
# [ ] plotting function?
# [ ] function to copy demo notebook and data

# [ ] check that groupcols are in the data, otherwise quit
# [ ] check that xname/yname/yerr are in the data, otherwise quit
# [ ] check that method is leastsq, otherwise quit

# [ ] make confidence interval calculation work without multithreading
# [ ] search for NaNs in model column and fill
# [ ] make unrecognized parameter format throw an error and quit
