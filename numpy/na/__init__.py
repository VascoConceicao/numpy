"""
=============
Named Arrays
=============

Named arrays extend NumPy arrays with support for named dimensions.

This submodule introduces the `NamedArray` class and helper functions
that allow you to work with arrays whose axes have semantic names
rather than relying on positional indexing alone. This improves
readability, reduces indexing errors, and enables more expressive
manipulation of multidimensional data.
"""

from . import core
from .core import *

__all__ = ['core']
__all__ += core.__all__

from numpy._pytesttester import PytestTester

test = PytestTester(__name__)
del PytestTester
