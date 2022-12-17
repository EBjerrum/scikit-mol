#%%
from multiprocessing import Pool, get_context
from rdkit import Chem
from rdkit import DataStructs
#from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdMHFPFingerprint

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix

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
        arr = np.zeros((self.nBits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def _transform_mol(self, mol):
        fp = self._mol2fp(mol)
        arr = self._fp2array(fp)
        return arr

    def fit(self, X, y=None):
        """Included for scikit-learn compatibility, does nothing"""
        return self


    def _transform(self, X):
        arr = np.zeros((len(X), self.nBits), dtype=np.int8)
        for i, mol in enumerate(X):
            arr[i,:] = self._transform_mol(mol)
        return arr

    def _transform_sparse(self, X):
        arr = lil_matrix((len(X), self.nBits), dtype=np.int8)
        for i, mol in enumerate(X):
            arr[i,:] = self._transform_mol(mol)
        return arr

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
        if not self.parallel:
            return self._transform(X)
        elif self.parallel:
            n_processes = self.parallel if self.parallel > 1 else None #Pool(processes=None) autodetects
            #with get_context("spawn").Pool(processes=n_processes) as pool:
                #print(f"{n_processes=}")
            pool = Pool(processes=n_processes)
            x_chunks = np.array_split(X, 4)
            #print(f"{x_chunks=}")
            arrays = pool.map(self._transform, x_chunks)
            # arrays = async_obj.get(20)
            pool.close()
            # print(f"{arrays=}")
            arr = np.concatenate(arrays)
            return arr
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


    @property
    def fpSize(self):
        return self.nBits

    #Scikit-Learn expects to be able to set fpSize directly on object via .set_params(), so this updates nBits used by the abstract class
    @fpSize.setter
    def fpSize(self, fpSize):
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

class AtomPairFingerprintTransformer(FpsTransformer): #FIXME, some of the init arguments seems to be molecule specific, and should probably not be setable?
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

class SECFingerprintTransformer(FpsTransformer):
    # https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0321-8
    def __init__(self, radius:int=3, rings:bool=True, isomeric:bool=False, kekulize:bool=False,
                 min_radius:int=1, length:int=2048, n_permutations:int=0, seed:int=0):
        """Transforms the RDKit mol into the SMILES extended connectivity fingerprint (SECFP)

        Args:
            radius (int, optional): The MHFP radius. Defaults to 3.
            rings (bool, optional): Whether or not to include rings in the shingling. Defaults to True.
            isomeric (bool, optional): Whether the isomeric SMILES to be considered. Defaults to False.
            kekulize (bool, optional): Whether or not to kekulize the extracted SMILES. Defaults to False.
            min_radius (int, optional): The minimum radius that is used to extract n-gram. Defaults to 1.
            length (int, optional): The length of the folded fingerprint. Defaults to 2048.
            n_permutations (int, optional): The number of permutations used for hashing. Defaults to 0.
            seed (int, optional): The value used to seed numpy.random. Defaults to 0.
        """
        self.radius = radius
        self.rings = rings
        self.isomeric = isomeric
        self.kekulize = kekulize
        self.min_radius = min_radius
        self.length = length
        self._n_permutations = n_permutations
        self._seed = seed
        # create the encoder instance
        self._recreate_encoder()

    def _mol2fp(self, mol):
        return self.secfp_encoder.EncodeSECFPMol(mol, self.radius, self.rings, self.isomeric, self.kekulize, self.min_radius, self.length) 

    def _recreate_encoder(self):
        self.secfp_encoder = rdMHFPFingerprint.MHFPEncoder(self._n_permutations, self._seed)

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        # each time the seed parameter is modified refresh an instace of the encoder
        self._recreate_encoder()

    @property
    def n_permutations(self):
        return self._n_permutations

    @n_permutations.setter
    def n_permutations(self, n_permutations):
        self._n_permutations = n_permutations
        # each time the n_permutations parameter is modified refresh an instace of the encoder
        self._recreate_encoder()

    @property
    def nBits(self):
        # to be compliant with the requirement of the base class
        return self.length

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
                useChirality=self.useChirality, useBondTypes=self.useBondTypes
            )
        else:
            return rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol,self.radius,nBits=self.nBits, useFeatures=self.useFeatures,
                useChirality=self.useChirality, useBondTypes=self.useBondTypes
            )
        

class SmilesToMol(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

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

