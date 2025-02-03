"""
Core functionality for scikit-mol.

Users of scikit-mol should not need to use this module directly.
Users who want to create their own transformers should use this module.
"""

import functools
from dataclasses import dataclass

import numpy as np
from packaging.version import Version

SKLEARN_VERSION_PANDAS_OUT = Version("1.2")

DEFAULT_MOL_COLUMN_NAME = "ROMol"


class NoFitNeededMixin:
    """
    Mixin class to add a __sklearn_is_fitted__ method to a transformes, which does not need to be fitted.
    """

    def __sklearn_is_fitted__(self):
        return True


@dataclass
class InvalidMol:
    """
    Represents molecules which raised an error during a pipeline step.
    Evaluates to False in boolean contexts.
    """

    pipeline_step: str
    error: str

    def __bool__(self):
        return False

    def __repr__(self):
        return f"InvalidMol('{self.pipeline_step}', error='{self.error}')"


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
        return X  # Flatt Arrays and list-like data are also supported #TODO, add a warning about non-2D data if logging is implemented
    if shape[1] != 1:
        raise ValueError(
            "Only one column supported. You may want to use a ColumnTransformer https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html "
        )
    return np.array(X).flatten()


def check_transform_input(method):
    """
    Decorator to check the input of the _transform method
    and make it compatible with the scikit-learn API and with downstream methods.
    """

    @functools.wraps(method)
    def wrapper(obj, X):
        X = _validate_transform_input(X)
        result = method(obj, X)
        # If the output of the _transform method
        # must be changed depending on the initial type of X, do it here.
        return result

    return wrapper


def feature_names_default_mol(method):
    """
    Decorator that returns the default feature names for the mol object
    """

    @functools.wraps(method)
    def wrapper(obj, input_features=None):
        prefix = DEFAULT_MOL_COLUMN_NAME
        if input_features is not None:
            return np.array([f"{prefix}_{name}" for name in input_features])
        else:
            return np.array([prefix])

    return wrapper
