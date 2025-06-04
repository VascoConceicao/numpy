from . import core
from .core import *

__all__ = ['core']
__all__ += core.__all__

from numpy._pytesttester import PytestTester

test = PytestTester(__name__)
del PytestTester
