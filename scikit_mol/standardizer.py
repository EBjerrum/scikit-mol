# A scikit-learn compatible molecule standardizer
# Author: Son Ha


import functools

import numpy as np
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.rdBase import BlockLogs
from sklearn.base import BaseEstimator, TransformerMixin

from scikit_mol.core import (
    InvalidMol,
    NoFitNeededMixin,
    check_transform_input,
    feature_names_default_mol,
)
from scikit_mol.parallel import parallelized_with_batches


class Standardizer(TransformerMixin, NoFitNeededMixin, BaseEstimator):
    """Input a list of rdkit mols, output the same list but standardised"""

    def __init__(self, neutralize=True, parallel=None, safe_inference_mode=False):
        self.neutralize = neutralize
        self.parallel = parallel
        self.safe_inference_mode = safe_inference_mode

    def fit(self, X, y=None):
        return self

    def _standardize_mol(self, mol):
        if not mol:
            if self.safe_inference_mode:
                if isinstance(mol, InvalidMol):
                    return mol
                else:
                    return InvalidMol(str(self), f"Invalid input molecule: {mol}")
            else:
                raise ValueError(f"Invalid input molecule: {mol}")

        try:
            block = BlockLogs()  # Block all RDkit logging
            # Normalizing functional groups
            clean_mol = rdMolStandardize.Cleanup(mol)
            # Get parents fragments
            parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
            # Neutralise
            if self.neutralize:
                uncharger = rdMolStandardize.Uncharger()
                uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
            else:
                uncharged_parent_clean_mol = parent_clean_mol
            del block  # Release logging block to previous state
            Chem.SanitizeMol(uncharged_parent_clean_mol)
            return uncharged_parent_clean_mol
        except Exception as e:
            if self.safe_inference_mode:
                return InvalidMol(str(self), f"Standardization failed: {e}")
            else:
                raise

    def _transform(self, X):
        return np.array([self._standardize_mol(mol) for mol in X]).reshape(-1, 1)

    @feature_names_default_mol
    def get_feature_names_out(self, input_features=None):
        return input_features

    @check_transform_input
    def transform(self, X, y=None):
        parameters = self.get_params()
        func = functools.partial(parallel_helper, self.__class__.__name__, parameters)
        arrays = parallelized_with_batches(func, X, self.parallel)
        arr = np.concatenate(arrays)
        return arr


def parallel_helper(classname, parameters, X_mols):
    """Parallel_helper takes a tuple with classname, the objects parameters and the mols to process.
    Then instantiates the class with the parameters and processes the mol.
    Intention is to be able to do this in child processes as some classes can't be pickled"""
    from scikit_mol import standardizer

    transformer = getattr(standardizer, classname)(**parameters)
    return transformer._transform(X_mols)
