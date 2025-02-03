# A scikit-learn compatible molecule standardizer
# Author: Son Ha


import functools
from typing import Optional

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
    """Standardize molecules with RDKit

    Parameters
    ----------
    neutralize : bool, optional
        If True, neutralizes the molecule, by default True
    n_jobs : Optional[int], optional
        The maximum number of concurrently running jobs.
        None is a marker for 'unset' that will be interpreted as n_jobs=1 unless the call is performed under a parallel_config() context manager that sets another value for n_jobs., by default None
    safe_inference_mode : bool, optional
        If True, enables safeguards for handling invalid data during inference.
        This should only be set to True when deploying models to production, by default False
    """

    def __init__(
        self,
        neutralize: bool = True,
        n_jobs: Optional[int] = None,
        safe_inference_mode: bool = False,
    ):
        self.neutralize = neutralize
        self.n_jobs = n_jobs
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
        arrays = parallelized_with_batches(func, X, self.n_jobs)
        arr = np.concatenate(arrays)
        return arr


def parallel_helper(classname, parameters, X_mols):
    """Parallel_helper takes a tuple with classname, the objects parameters and the mols to process.
    Then instantiates the class with the parameters and processes the mol.
    Intention is to be able to do this in child processes as some classes can't be pickled"""
    from scikit_mol import standardizer

    transformer = getattr(standardizer, classname)(**parameters)
    return transformer._transform(X_mols)
