# A scikit-learn compatible molecule standardizer
# Author: Son Ha

from rdkit import Chem
from sklearn.base import BaseEstimator, TransformerMixin
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.rdBase import BlockLogs


class Standardizer(BaseEstimator, TransformerMixin):
    """ Input a list of rdkit mols, output the same list but standardised 
    """
    def __init__(self, neutralize=True):
        self.neutralize = neutralize

    def fit(self, X, y=None):
        return self        
        
    def transform(self, X):
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
            # Add to final list
            arr.append(uncharged_parent_clean_mol)
        
        del block # Release logging block to previous state
        return(arr)
