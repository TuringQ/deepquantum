"""
This is the top level module from which all basic functions and classes of
DeepQuantum can be directly imported.
"""

from importlib.metadata import version
__version__ = version('deepquantum')

from .operation import *
from .qmath import *
from .gate import *
from .layer import *
from .state import *
from .circuit import *
from .ansatz import *
from .utils import *
from . import photonic
