from .functions import *
from .pdLSR import pdLSR

from .docstring import DOCSTRING
from .version import VERSION

__doc__ = DOCSTRING
__version__ = VERSION


# TODOs:
# [ ] make confidence interval calculation work without multithreading
# [ ] add statsmodels and maybe scikit-learn functionality
# [ ] add .sortlevel(axis=0) to all tables?
# [ ] function to copy demo notebook and data for demo?
# [ ] seaborn factgrid plotting function?
