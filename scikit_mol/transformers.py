#%%
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

#%%
class MorganTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nBits=2048, radius=2, useChirality=False, useBondTypes=True, useFeatures=False, useCounts=False):
        self.nBits = nBits
        self.radius = radius
        self.useChirality = useChirality
        self.useBondTypes = useBondTypes
        self.useFeatures = useFeatures
        self.useCounts = useCounts

    def _mol2fp(self, mol):
        if self.useCounts:
            return GetHashedMorganFingerprint(
                mol,self.radius,nBits=self.nBits, useFeatures=self.useFeatures,
                useChirality=self.useChirality,
            )
        else:
            return GetMorganFingerprintAsBitVect(
                mol,self.radius,nBits=self.nBits, useFeatures=self.useFeatures,
                useChirality=self.useChirality,
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