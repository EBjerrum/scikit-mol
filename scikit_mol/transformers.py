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
        """Included for scikit-learn compatibility, does nothing"""
        return self

    def transform(self, X, y=None):
        """Transform a list of RDKit molecule objects into a fingerprint array

        Parameters
        ----------
        X : (List, np.array, pd.Series)
            A list of RDKit molecules
        y : NoneType, optional
            Target values for scikit-learn compatibility, not used, by default None

        Returns
        -------
        np.array
            Fingerprints, shape (samples, fingerprint size)
        """
        arr = np.zeros((len(X), self.nBits))
        for i, mol in enumerate(X):
            arr[i,:] = self._transform_mol(mol)
        return arr


class MACCSTransformer(FpsTransformer):
    def __init__(self):
        """MACCS keys fingerprinter
        calculates the 167 fixed MACCS keys
        """
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
        """Calculates the RDKit fingerprints

        Parameters
        ----------
        minPath : int, optional
            the minimum path length (in bonds) to be included, by default 1
        maxPath : int, optional
            the maximum path length (in bonds) to be included, by default 7
        useHs : bool, optional
            toggles inclusion of Hs in paths (if the molecule has explicit Hs), by default True
        branchedPaths : bool, optional
            toggles generation of branched subgraphs, not just linear paths, by default True
        useBondOrder : bool, optional
            toggles inclusion of bond orders in the path hashes, by default True
        countSimulation : bool, optional
            if set, use count simulation while generating the fingerprint, by default False
        countBounds : _type_, optional
            boundaries for count simulation, corresponding bit will be set if the count is higher than the number provided for that spot, by default None
        fpSize : int, optional
            size of the generated fingerprint, does not affect the sparse versions, by default 2048
        numBitsPerFeature : int, optional
            the number of bits set per path/subgraph found, by default 2
        atomInvariantsGenerator : _type_, optional
            atom invariants to be used during fingerprint generation, by default None
        """
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
            return rdMolDescriptors.GetHashedAtomPairFingerprint(mol, nBits=self,
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
        """Transform RDKit mols into Count or bit-based hashed MorganFingerprints

        Parameters
        ----------
        nBits : int, optional
            Size of the hashed fingerprint, by default 2048
        radius : int, optional
            Radius of the fingerprint, by default 2
        useChirality : bool, optional
            Include chirality in calculation of the fingerprint keys, by default False
        useBondTypes : bool, optional
            Include bondtypes in calculation of the fingerprint keys, by default True
        useFeatures : bool, optional
            use chemical features, rather than atom-type in calculation of the fingerprint keys, by default False
        useCounts : bool, optional
            If toggled will create the count and not bit-based fingerprint, by default False
        """
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
        """Included for scikit-learn compatibility, does nothing"""
        return self

    def transform(self, X_smiles_list):
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
            Raises ValueError of a SMILES string is unparsable by RDKit
        """
        X_out = []

        for smiles in X_smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                X_out.append(mol)
            else:
                raise ValueError(f'Issue with parsing SMILES {smiles}\nYou probably should use the scikit-mol.sanitizer.Sanitizer first')

        return X_out
