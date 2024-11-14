# A scikit-learn compatible molecule standardizer
# Author: Son Ha

import multiprocessing
from rdkit import Chem
from sklearn.base import BaseEstimator, TransformerMixin
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.rdBase import BlockLogs
import numpy as np

from scikit_mol.core import check_transform_input, feature_names_default_mol, InvalidMol


class Standardizer(BaseEstimator, TransformerMixin):
    """Input a list of rdkit mols, output the same list but standardised"""

    def __init__(self, neutralize=True, parallel=False, safe_inference_mode=False):
        self.neutralize = neutralize
        self.parallel = parallel
        self.start_method = None  # TODO implement handling of start_method
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
                return InvalidMol(str(self), f"Standardization failed: {str(e)}")
            else:
                raise

    def _transform(self, X):
        return np.array([self._standardize_mol(mol) for mol in X]).reshape(-1, 1)

    @feature_names_default_mol
    def get_feature_names_out(self, input_features=None):
        return input_features

    @check_transform_input
    def transform(self, X, y=None):
        if not self.parallel:
            return self._transform(X)

        elif self.parallel:
            n_processes = (
                self.parallel if self.parallel > 1 else None
            )  # Pool(processes=None) autodetects
            n_chunks = (
                n_processes * 2
                if n_processes is not None
                else multiprocessing.cpu_count() * 2
            )  # TODO, tune the number of chunks per child process

            with multiprocessing.get_context(self.start_method).Pool(
                processes=n_processes
            ) as pool:
                x_chunks = np.array_split(X, n_chunks)
                parameters = self.get_params()
                arrays = pool.map(
                    parallel_helper,
                    [
                        (self.__class__.__name__, parameters, x_chunk)
                        for x_chunk in x_chunks
                    ],
                )
                arr = np.concatenate(arrays)
            return arr


def parallel_helper(args):
    """Parallel_helper takes a tuple with classname, the objects parameters and the mols to process.
    Then instantiates the class with the parameters and processes the mol.
    Intention is to be able to do this in child processes as some classes can't be pickled"""
    classname, parameters, X_mols = args
    from scikit_mol import standardizer

    transformer = getattr(standardizer, classname)(**parameters)
    return transformer._transform(X_mols)
