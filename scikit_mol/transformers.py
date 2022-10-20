#%%
from rdkit import Chem
from rdkit import DataStructs
#from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from abc import ABC, abstractmethod

#%%
class FpsTransformer(ABC, BaseEstimator, TransformerMixin):
    @abstractmethod
    def _mol2fp(self, mol):
        """Generate descriptor from mol

        MUST BE OVERWRITTEN
        """
        raise NotImplementedError("_mol2fp not implemented")

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


class MACCSTransformer(FpsTransformer):
    def __init__(self):
        self.nBits = 167
        pass

    def _mol2fp(self, mol):
        return rdMolDescriptors.GetMACCSKeysFingerprint(
            mol
        )

class RDKitFPTransformer(FpsTransformer):
    def __init__(self, minPath:int = 1, maxPath:int =7, useHs:bool = True, branchedPaths:bool = True,
                 useBondOrder:bool = True, countSimulation:bool = False, countBounds = None,
                 fpSize:int  = 2048, numBitsPerFeature:int = 2, atomInvariantsGenerator = None
                 ):
        self.minPath = minPath
        self.maxPath = maxPath
        self.useHs = useHs
        self.branchedPaths = branchedPaths
        self.useBondOrder = useBondOrder
        self.countSimulation = countSimulation
        self.countBounds = countBounds
        self.fpSize = fpSize
        self.numBitsPerFeature = numBitsPerFeature
        self.atomInvariantsGenerator = atomInvariantsGenerator
        self.nBits = fpSize

    def _mol2fp(self, mol):
        generator = rdFingerprintGenerator.GetRDKitFPGenerator(minPath=self.minPath, maxPath=self.maxPath,
                                                               useHs=self.useHs, branchedPaths=self.branchedPaths,
                                                               useBondOrder=self.useBondOrder,
                                                               countSimulation=self.countSimulation,
                                                               countBounds=self.countBounds, fpSize=self.fpSize,
                                                               numBitsPerFeature=self.numBitsPerFeature,
                                                               atomInvariantsGenerator=self.atomInvariantsGenerator
                                                               )
        return generator.GetFingerprint(mol)

class AtomPairFingerprintTransformer(FpsTransformer):
    def __init__(self, minLength:int = 1, maxLength:int = 30, fromAtoms = 0, ignoreAtoms = 0, atomInvariants = 0,
                 nBitsPerEntry:int = 4, includeChirality:bool = False, use2D:bool = True, confId:int = -1, nBits=2048,
                 useCounts:bool=False):
        self.minLength = minLength
        self.maxLength = maxLength
        self.fromAtoms = fromAtoms
        self.ignoreAtoms = ignoreAtoms
        self.atomInvariants = atomInvariants
        self.includeChirality = includeChirality
        self.use2D = use2D
        self.confId = confId
        self.nBits = nBits
        self.nBitsPerEntry = nBitsPerEntry
        self.useCounts = useCounts

    def _mol2fp(self, mol):
        if self.useCounts:
            return rdMolDescriptors.GetHashedAtomPairFingerprint(mol, nBits=self.nBits,
                                                                 minLength=self.minLength,
                                                                 maxLength=self.maxLength,
                                                                 fromAtoms=self.fromAtoms,
                                                                 ignoreAtoms=self.ignoreAtoms,
                                                                 atomInvariants=self.atomInvariants,
                                                                 includeChirality=self.includeChirality,
                                                                 use2D=self.use2D,
                                                                 confId=self.confId
                                                           )
        else:
            return rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=self.nBits,
                                                                          minLength=self.minLength,
                                                                          maxLength=self.maxLength,
                                                                          fromAtoms=self.fromAtoms,
                                                                          ignoreAtoms=self.ignoreAtoms,
                                                                          atomInvariants=self.atomInvariants,
                                                                          nBitsPerEntry=self.nBitsPerEntry,
                                                                          includeChirality=self.includeChirality,
                                                                          use2D=self.use2D,
                                                                          confId=self.confId
                                                       )

class TopologicalTorsionFingerprintTransformer(FpsTransformer):
    def __init__(self, targetSize:int = 4, fromAtoms = 0, ignoreAtoms = 0, atomInvariants = 0,
                 includeChirality:bool = False, nBitsPerEntry:int = 4, nBits=2048,
                 useCounts:bool=False):
        self.targetSize = targetSize
        self.fromAtoms = fromAtoms
        self.ignoreAtoms = ignoreAtoms
        self.atomInvariants = atomInvariants
        self.includeChirality = includeChirality
        self.nBitsPerEntry = nBitsPerEntry
        self.nBits = nBits
        self.useCounts = useCounts

    def _mol2fp(self, mol):
        if self.useCounts:
            return rdMolDescriptors.GetHashedTopologicalTorsionFingerprint(mol, nBits=self.nBits,
                                                                           targetSize=self.targetSize,
                                                                           fromAtoms=self.fromAtoms,
                                                                           ignoreAtoms=self.ignoreAtoms,
                                                                           atomInvariants=self.atomInvariants,
                                                                           includeChirality=self.includeChirality,
                                                           )
        else:
            return rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=self.nBits,
                                                                                    targetSize=self.targetSize,
                                                                                    fromAtoms=self.fromAtoms,
                                                                                    ignoreAtoms=self.ignoreAtoms,
                                                                                    atomInvariants=self.atomInvariants,
                                                                                    includeChirality=self.includeChirality,
                                                                                    nBitsPerEntry=self.nBitsPerEntry
                                                                                    )


class MorganTransformer(FpsTransformer):
    def __init__(self, nBits=2048, radius=2, useChirality=False, useBondTypes=True, useFeatures=False, useCounts=False):
        self.nBits = nBits
        self.radius = radius
        self.useChirality = useChirality
        self.useBondTypes = useBondTypes
        self.useFeatures = useFeatures
        self.useCounts = useCounts

    def _mol2fp(self, mol):
        if self.useCounts:
            return rdMolDescriptors.GetHashedMorganFingerprint(
                mol,self.radius,nBits=self.nBits, useFeatures=self.useFeatures,
                useChirality=self.useChirality,
            )
        else:
            return rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol,self.radius,nBits=self.nBits, useFeatures=self.useFeatures,
                useChirality=self.useChirality,
            )
        

class SmilesToMol(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        #Nothing to do here
        return self

    def transform(self, X_smiles_list):
        X_out = []

        for smiles in X_smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                X_out.append(mol)
            else:
                raise ValueError(f'Issue with parsing SMILES {smiles}\nYou probably should use the scikit-mol.sanitizer.Sanitizer first')

        return X_out
