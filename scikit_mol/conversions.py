from multiprocessing import get_context
import multiprocessing
from typing import Union
from rdkit import Chem

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from scikit_mol.core import (
    check_transform_input,
    feature_names_default_mol,
    DEFAULT_MOL_COLUMN_NAME,
    InvalidMol,
)

# from scikit_mol._invalid import InvalidMol


class SmilesToMolTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, parallel: Union[bool, int] = False):
        self.parallel = parallel
        self.start_method = None  # TODO implement handling of start_method

    @feature_names_default_mol
    def get_feature_names_out(self, input_features=None):
        return input_features

    def fit(self, X=None, y=None):
        """Included for scikit-learn compatibility, does nothing"""
        return self

    def transform(self, X_smiles_list, y=None):
        """Converts SMILES into RDKit mols

        Parameters
        ----------
        X_smiles_list : list-like
            A list of RDKit parsable strings

        Returns
        -------
        List
            List of RDKit mol objects

        Raises
        ------
        ValueError
            Raises ValueError if a SMILES string is unparsable by RDKit
        """

        if not self.parallel:
            return self._transform(X_smiles_list)
        elif self.parallel:
            n_processes = (
                self.parallel if self.parallel > 1 else None
            )  # Pool(processes=None) autodetects
            n_chunks = (
                n_processes * 2
                if n_processes is not None
                else multiprocessing.cpu_count() * 2
            )  # TODO, tune the number of chunks per child process
            with get_context(self.start_method).Pool(processes=n_processes) as pool:
                x_chunks = np.array_split(X_smiles_list, n_chunks)
                arrays = pool.map(
                    self._transform, x_chunks
                )  # is the helper function a safer way of handling the picklind and child process communication
                arr = np.concatenate(arrays)
                return arr

    @check_transform_input
    def _transform(self, X):
        X_out = []
        for smiles in X:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                X_out.append(mol)
            else:
                mol = Chem.MolFromSmiles(smiles, sanitize=False)
                if mol:
                    errors = Chem.DetectChemistryProblems(mol)
                    error_message = "\n".join(error.Message() for error in errors)
                    message = f"Invalid SMILES: {error_message}"
                else:
                    message = f"Invalid SMILES: {smiles}"
                X_out.append(InvalidMol(str(self), message))
        return X_out

    @check_transform_input
    def inverse_transform(
        self, X_mols_list, y=None
    ):  # TODO, maybe the inverse transform should be configurable e.g. isomericSmiles etc.?
        X_out = []

        for mol in X_mols_list:
            if mol:
                try:
                    smiles = Chem.MolToSmiles(mol)
                    X_out.append(smiles)
                except Exception as e:
                    X_out.append(InvalidMol(str(self), str(e)))
            else:
                X_out.append(InvalidMol(str(self), f"Not a Mol: {mol}"))

        return np.array(X_out).reshape(-1, 1)
