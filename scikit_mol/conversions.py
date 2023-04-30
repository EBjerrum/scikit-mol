from multiprocessing import get_context
import multiprocessing
from typing import Union
from rdkit import Chem

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SmilesToMolTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, parallel: Union[bool, int] = False):
        self.parallel = parallel
        self.start_method = None  #TODO implement handling of start_method

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
            n_processes = self.parallel if self.parallel > 1 else None # Pool(processes=None) autodetects
            n_chunks = n_processes*2 if n_processes is not None else multiprocessing.cpu_count()*2 #TODO, tune the number of chunks per child process
            with get_context(self.start_method).Pool(processes=n_processes) as pool:
                    x_chunks = np.array_split(X_smiles_list, n_chunks) 
                    arrays = pool.map(self._transform, x_chunks) #is the helper function a safer way of handling the picklind and child process communication

                    arr = np.concatenate(arrays)
                    return arr

    def _transform(self, X_smiles_list):
        X_out = []
        for smiles in X_smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                X_out.append(mol)
            else:
                raise ValueError(f'Issue with parsing SMILES {smiles}\nYou probably should use the scikit-mol.sanitizer.Sanitizer on your dataset first')

        return X_out

    def inverse_transform(self, X_mols_list, y=None): #TODO, maybe the inverse transform should be configurable e.g. isomericSmiles etc.?
        X_out = []

        for mol in X_mols_list:
            smiles = Chem.MolToSmiles(mol)
            X_out.append(smiles)

        return X_out
