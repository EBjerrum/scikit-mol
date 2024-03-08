"""
Core functionality for scikit-mol.

Users of scikit-mol should not need to use this module directly.
Users who want to create their own transformers should use this module.
"""

import functools
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

@dataclass
class PythonVersion:
    """
    Simple class to represent a Python version.
    """
    major: int
    minor: int
    patch: Optional[int] = None

    def __str__(self):
        """The string representation of the Python version."""
        if self.patch is not None:
            return f"{self.major}.{self.minor}.{self.patch}"
        else:
            return f"{self.major}.{self.minor}"

    def as_tuple(self):
        """The Python version as a tuple."""
        if self.patch is not None:
            return (self.major, self.minor, self.patch)
        else:
            return (self.major, self.minor)

MIN_PYTHON_FOR_PANDAS_OUT = PythonVersion(3, 8)


def _validate_transform_input(X):
        """Validate and adapt the input of the _transform method"""
        try:
            shape = X.shape
        except AttributeError:
            # If X is not array-like or dataframe-like,
            # we just return it as is, so users can use simple lists and sequences.
            return X
        # If X is an array-like or dataframe-like, we make sure it is compatible with
        # the scikit-learn API, and that it contains a single column:
        # scikit-mol transformers need a single column with smiles or mols.
        if len(shape) == 1:
            return X # Flatt Arrays and list-like data are also supported #TODO, add a warning about non-2D data if logging is implemented
        if shape[1] != 1:
            raise ValueError("Only one column supported. You may want to use a ColumnTransformer https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html ")
        return np.array(X).flatten()

def check_transform_input(method):
    """
    Decorator to check the input of the _transform method
    and make it compatible with the scikit-learn API and with downstream methods.
    """
    @functools.wraps(method)
    def wrapper(obj, X):
        X = _validate_transform_input(X)
        result =  method(obj, X)
        # If the output of the _transform method
        # must be changed depending on the initial type of X, do it here.
        return result

    return wrapper