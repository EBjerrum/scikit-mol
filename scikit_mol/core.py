"""
Core functionality for scikit-mol.

Users of scikit-mol should not need to use this module directly.
Users who want to create their own transformers should use this module.
"""

import pandas as pd

def get_mols_from_X(X):
        """Get the molecules iterable from the input X"""
        if isinstance(X, pd.DataFrame):
            try:
                # TODO: possibly handle the case in which a DataFrame
                # contains multiple molecule columns (as if from a column selector transformer).
                # In that case, the resulting array should be a concatenation of the fingerprint arrays
                # for each molecule column.
                # TODO: Change core logic of how scikit-mol transformers handle input:
                # make them only accept 2D arrays with a single column (and possibly flat lists).
                # See GitHub discussion.
                return X.loc[:, "ROMol"]
            except KeyError:
                return X.iloc[:, 0]
        else:
            return X