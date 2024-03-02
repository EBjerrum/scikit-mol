"""
Core functionality for scikit-mol.

Users of scikit-mol should not need to use this module directly.
Users who want to create their own transformers should use this module.
"""

import functools

import pandas as pd

def _get_mols_from_X(X):
        """Get the molecules iterable from the input X"""
        if isinstance(X, pd.DataFrame):
            try:
                # TODO: Change core logic of how scikit-mol transformers handle input:
                # make them only accept 2D arrays with a single column (and possibly flat lists).
                # See GitHub discussion.
                return X.loc[:, "ROMol"]
            except KeyError:
                return X.iloc[:, 0]
        else:
            return X

def check_transform_input(method):
    """
    Decorator to check the input of the _transform method
    and make it compatible with the scikit-learn API and with downstream methods.
    """
    @functools.wraps(method)
    def wrapper(obj, X):
        X = _get_mols_from_X(X)
        result =  method(obj, X)
        # If the output of the _transform method
        # must be changed depending on the initial type of X, do it here.
        return result

    return wrapper