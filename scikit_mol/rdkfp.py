#%%
from ctypes.wintypes import MAX_PATH
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.AllChem import RDKFingerprint

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
#%%
class RDfpTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, minPath=1,maxPath=7,fpSize=2048,nBitsPerHash=2,useHs=True,tgtDensity=0.0,minSize=128,
    branchedPaths=True,useBondOrder=True,atomInvariants=0,fromAtoms=0,atomBits=None):
        self.minPath = minPath
        self.maxPath = maxPath
        self.fpSize = fpSize
        self.nBitsPerHash = nBitsPerHash
        self.useHs = useHs
        self.tgtDensity = tgtDensity
        self.minSize = minSize
        self.branchedPaths = branchedPaths
        self.useBondOrder = useBondOrder
        self.atomInvariants = atomInvariants
        self.fromAtoms = fromAtoms
        self.atomBits = atomBits

    def _mol2fp(self, mol):
        return RDKFingerprint(mol,minPath=self.minPath,maxPath=self.maxPath,fpSize=self.fpSize,
        nBitsPerHash=self.nBitsPerHash,useHs=self.useHs,tgtDensity=self.tgtDensity,minSize=self.minSize,
        branchedPaths=self.branchedPaths,useBondOder=self.useBondOrder,atomInvariants=self.atomInvariants,
        fromAtoms=self.fromAtoms,atomBits=self.atomBits,
        )

    def _fp2array(self, fp):
        arr = np.zeros((self.nBits,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def _transform_mol(self, mol):
        fp = self._mol2fp(mol)
        arr = self._fp2array(fp)
        return arr

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        arr = np.zeros((len(X), self.nBits))
        for i, mol in enumerate(X):
            arr[i,:] = self._transform_mol(mol)
        return arr

        
# %%
