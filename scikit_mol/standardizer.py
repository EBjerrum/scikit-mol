# A scikit-learn compatible molecule standardizer
# Author: Son Ha

import multiprocessing
from rdkit import Chem
from sklearn.base import BaseEstimator, TransformerMixin
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.rdBase import BlockLogs
import numpy as np


class Standardizer(BaseEstimator, TransformerMixin):
    """ Input a list of rdkit mols, output the same list but standardised 
    """
    def __init__(self, neutralize=True, parallel=False):
        self.neutralize = neutralize
        self.parallel = parallel
        self.start_method = None #TODO implement handling of start_method

    def fit(self, X, y=None):
        return self        
        
    def _transform(self, X):
        block = BlockLogs() # Block all RDkit logging
        arr = []
        for mol in X:
            # Normalizing functional groups
            # https://molvs.readthedocs.io/en/latest/guide/standardize.html
            clean_mol = rdMolStandardize.Cleanup(mol) 
            # Get parents fragments
            parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
            # Neutralise
            if self.neutralize:
                uncharger = rdMolStandardize.Uncharger()
                uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
            else:
                uncharged_parent_clean_mol = parent_clean_mol
            # Add to final list
            arr.append(uncharged_parent_clean_mol)
        
        del block # Release logging block to previous state
        return(arr)
    
    def transform(self, X, y=None):
        if not self.parallel:
            return self._transform(X)

        elif self.parallel:
            n_processes = self.parallel if self.parallel > 1 else None # Pool(processes=None) autodetects
            n_chunks = n_processes*2 if n_processes is not None else multiprocessing.cpu_count()*2 #TODO, tune the number of chunks per child process
            
            with multiprocessing.get_context(self.start_method).Pool(processes=n_processes) as pool:
                x_chunks = np.array_split(X, n_chunks)
                #TODO check what is fastest, pickle or recreate and do this only for classes that need this
                #arrays = pool.map(self._transform, x_chunks)
                parameters = self.get_params()
                arrays = pool.map(parallel_helper, [(self.__class__.__name__, parameters, x_chunk) for x_chunk in x_chunks]) 
                arr = np.concatenate(arrays)
            return arr



def parallel_helper(args):
    """Parallel_helper takes a tuple with classname, the objects parameters and the mols to process.
    Then instantiates the class with the parameters and processes the mol.
    Intention is to be able to do this in chilcprocesses as some classes can't be pickled"""
    classname, parameters, X_mols = args
    from scikit_mol import standardizer
    transformer = getattr(standardizer, classname)(**parameters)
    return transformer._transform(X_mols)
