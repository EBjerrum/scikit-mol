#%%
from multiprocessing import Pool, get_context
import multiprocessing
import re
from typing import Union
from rdkit import Chem
from rdkit import DataStructs
#from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdMHFPFingerprint
from rdkit.Avalon import pyAvalonTools

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from scipy.sparse import vstack

from sklearn.base import BaseEstimator, TransformerMixin
from scikit_mol.core import check_transform_input

from abc import ABC, abstractmethod

_PATTERN_FINGERPRINT_TRANSFORMER = re.compile(r"^(?P<fingerprint_name>\w+)FingerprintTransformer$")

#%%
class FpsTransformer(ABC, BaseEstimator, TransformerMixin):

    def __init__(self, parallel: Union[bool, int] = False, start_method: str = None):
        self.parallel = parallel
        self.start_method = start_method #TODO implement handling of start_method

    # The dtype of the fingerprint array computed by the transformer
    # If needed this property can be overwritten in the child class.
    _DTYPE_FINGERPRINT = np.int8

    def _get_column_prefix(self) -> str:
        matched = _PATTERN_FINGERPRINT_TRANSFORMER.match(type(self).__name__)
        if matched:
            fingerprint_name = matched.group("fingerprint_name")
            return f"fp_{fingerprint_name.lower()}"
        else:
            return "fp"

    def _get_n_digits_column_suffix(self) -> int:
        return len(str(self.nBits))

    def get_feature_names_out(self, input_features=None):
        prefix = self._get_column_prefix()
        n_digits = self._get_n_digits_column_suffix()
        return np.array([f"{prefix}_{str(i).zfill(n_digits)}" for i in range(1, self.nBits + 1)])

    @abstractmethod
    def _mol2fp(self, mol):
        """Generate descriptor from mol

        MUST BE OVERWRITTEN
        """
        raise NotImplementedError("_mol2fp not implemented")

    def _fp2array(self, fp):
        arr = np.zeros((self.nBits,), dtype=self._DTYPE_FINGERPRINT)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def _transform_mol(self, mol):
        fp = self._mol2fp(mol)
        arr = self._fp2array(fp)
        return arr

    def fit(self, X, y=None):
        """Included for scikit-learn compatibility

        Also sets the column prefix for use by the transform method with dataframe output.
        """
        return self

    @check_transform_input
    def _transform(self, X):
        arr = np.zeros((len(X), self.nBits), dtype=self._DTYPE_FINGERPRINT)
        for i, mol in enumerate(X):
            arr[i,:] = self._transform_mol(mol)
        return arr

    def _transform_sparse(self, X):
        arr = np.zeros((len(X), self.nBits), dtype=self._DTYPE_FINGERPRINT)
        for i, mol in enumerate(X):
            arr[i,:] = self._transform_mol(mol)
        
        return lil_matrix(arr)

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
            n_processes = self.parallel if self.parallel > 1 else None # Pool(processes=None) autodetects
            n_chunks = n_processes if n_processes is not None else multiprocessing.cpu_count() 
            
            with get_context(self.start_method).Pool(processes=n_processes) as pool:
                x_chunks = np.array_split(X, n_chunks)
                #TODO check what is fastest, pickle or recreate and do this only for classes that need this
                #arrays = pool.map(self._transform, x_chunks)
                parameters = self.get_params()
                # TODO: create "transform_parallel" function in the core module,
                # and use it here and in the descriptors transformer
                #x_chunks = [np.array(x).reshape(-1, 1) for x in x_chunks]
                arrays = pool.map(parallel_helper, [(self.__class__.__name__, parameters, x_chunk) for x_chunk in x_chunks]) 

                arr = np.concatenate(arrays)
            return arr


class MACCSKeysFingerprintTransformer(FpsTransformer):
    def __init__(self, parallel: Union[bool, int] = False):
        """MACCS keys fingerprinter
        calculates the 167 fixed MACCS keys
        """
        super().__init__(parallel = parallel)
        self.nBits = 167

    @property
    def nBits(self):
        return self._nBits

    @nBits.setter
    def nBits(self, nBits):
        if nBits != 167:
            raise ValueError("nBits can only be 167, matching the number of defined MACCS keys!")
        self._nBits = nBits

    def _mol2fp(self, mol):
        return rdMolDescriptors.GetMACCSKeysFingerprint(
            mol
        )

class RDKitFingerprintTransformer(FpsTransformer):
    def __init__(self, minPath:int = 1, maxPath:int =7, useHs:bool = True, branchedPaths:bool = True,
                 useBondOrder:bool = True, countSimulation:bool = False, countBounds = None,
                 fpSize:int  = 2048, numBitsPerFeature:int = 2, atomInvariantsGenerator = None,
                 parallel: Union[bool, int] = False
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
        super().__init__(parallel = parallel)
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
        generator = rdFingerprintGenerator.GetRDKitFPGenerator(minPath=int(self.minPath), maxPath=int(self.maxPath),
                                                               useHs=bool(self.useHs), branchedPaths=bool(self.branchedPaths),
                                                               useBondOrder=bool(self.useBondOrder),
                                                               countSimulation=bool(self.countSimulation),
                                                               countBounds=bool(self.countBounds), fpSize=int(self.fpSize),
                                                               numBitsPerFeature=int(self.numBitsPerFeature),
                                                               atomInvariantsGenerator=self.atomInvariantsGenerator
                                                               )
        return generator.GetFingerprint(mol)

class AtomPairFingerprintTransformer(FpsTransformer): #FIXME, some of the init arguments seems to be molecule specific, and should probably not be setable?
    def __init__(self, minLength:int = 1, maxLength:int = 30, fromAtoms = 0, ignoreAtoms = 0, atomInvariants = 0,
                 nBitsPerEntry:int = 4, includeChirality:bool = False, use2D:bool = True, confId:int = -1, nBits=2048,
                 useCounts:bool=False, parallel: Union[bool, int] = False,):
        super().__init__(parallel = parallel)
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
            return rdMolDescriptors.GetHashedAtomPairFingerprint(mol, nBits=int(self.nBits),
                                                                 minLength=int(self.minLength),
                                                                 maxLength=int(self.maxLength),
                                                                 fromAtoms=self.fromAtoms,
                                                                 ignoreAtoms=self.ignoreAtoms,
                                                                 atomInvariants=self.atomInvariants,
                                                                 includeChirality=bool(self.includeChirality),
                                                                 use2D=bool(self.use2D),
                                                                 confId=int(self.confId)
                                                           )
        else:
            return rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=int(self.nBits),
                                                                          minLength=int(self.minLength),
                                                                          maxLength=int(self.maxLength),
                                                                          fromAtoms=self.fromAtoms,
                                                                          ignoreAtoms=self.ignoreAtoms,
                                                                          atomInvariants=self.atomInvariants,
                                                                          nBitsPerEntry=int(self.nBitsPerEntry),
                                                                          includeChirality=bool(self.includeChirality),
                                                                          use2D=bool(self.use2D),
                                                                          confId=int(self.confId)
                                                       )

class TopologicalTorsionFingerprintTransformer(FpsTransformer):
    def __init__(self, targetSize:int = 4, fromAtoms = 0, ignoreAtoms = 0, atomInvariants = 0,
                 includeChirality:bool = False, nBitsPerEntry:int = 4, nBits=2048,
                 useCounts:bool=False, parallel: Union[bool, int] = False):
        super().__init__(parallel = parallel)
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
            return rdMolDescriptors.GetHashedTopologicalTorsionFingerprint(mol, nBits=int(self.nBits),
                                                                           targetSize=int(self.targetSize),
                                                                           fromAtoms=self.fromAtoms,
                                                                           ignoreAtoms=self.ignoreAtoms,
                                                                           atomInvariants=self.atomInvariants,
                                                                           includeChirality=bool(self.includeChirality),
                                                           )
        else:
            return rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=int(self.nBits),
                                                                                    targetSize=int(self.targetSize),
                                                                                    fromAtoms=self.fromAtoms,
                                                                                    ignoreAtoms=self.ignoreAtoms,
                                                                                    atomInvariants=self.atomInvariants,
                                                                                    includeChirality=bool(self.includeChirality),
                                                                                    nBitsPerEntry=int(self.nBitsPerEntry)
                                                                                    )

class MHFingerprintTransformer(FpsTransformer):
    # https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0321-8
    def __init__(self, radius:int=3, rings:bool=True, isomeric:bool=False, kekulize:bool=False,
                 min_radius:int=1, n_permutations:int=2048, seed:int=42, parallel: Union[bool, int] = False,):
        """Transforms the RDKit mol into the MinHash fingerprint (MHFP)

        Args:
            radius (int, optional): The MHFP radius. Defaults to 3.
            rings (bool, optional): Whether or not to include rings in the shingling. Defaults to True.
            isomeric (bool, optional): Whether the isomeric SMILES to be considered. Defaults to False.
            kekulize (bool, optional): Whether or not to kekulize the extracted SMILES. Defaults to False.
            min_radius (int, optional): The minimum radius that is used to extract n-gram. Defaults to 1.
            n_permutations (int, optional): The number of permutations used for hashing. Defaults to 0, 
            this is effectively the length of the FP
            seed (int, optional): The value used to seed numpy.random. Defaults to 0.
        """
        super().__init__(parallel = parallel)
        self.radius = radius
        self.rings = rings
        self.isomeric = isomeric
        self.kekulize = kekulize
        self.min_radius = min_radius
        #Set the .n_permutations and .seed without creating the encoder twice
        self._n_permutations = n_permutations
        self._seed = seed
        # create the encoder instance
        self._recreate_encoder()

    def __getstate__(self):
        # Get the state of the parent class
        state = super().__getstate__()
        # Remove the unpicklable property from the state
        state.pop("mhfp_encoder", None) # mhfp_encoder is not picklable
        return state

    def __setstate__(self, state):
        # Restore the state of the parent class
        super().__setstate__(state)
        # Re-create the unpicklable property
        self._recreate_encoder()

    _DTYPE_FINGERPRINT = np.int32

    def _mol2fp(self, mol):
        fp = self.mhfp_encoder.EncodeMol(mol, self.radius, self.rings, self.isomeric, self.kekulize, self.min_radius)
        return fp
    
    def _fp2array(self, fp):
        return np.array(fp)

    def _recreate_encoder(self):
        self.mhfp_encoder = rdMHFPFingerprint.MHFPEncoder(self._n_permutations, self._seed)

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        # each time the seed parameter is modified refresh an instance of the encoder
        self._recreate_encoder()

    @property
    def n_permutations(self):
        return self._n_permutations

    @n_permutations.setter
    def n_permutations(self, n_permutations):
        self._n_permutations = n_permutations
        # each time the n_permutations parameter is modified refresh an instance of the encoder
        self._recreate_encoder()

    @property
    def nBits(self):
        # to be compliant with the requirement of the base class
        return self._n_permutations

class SECFingerprintTransformer(FpsTransformer):
    # https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0321-8
    def __init__(self, radius:int=3, rings:bool=True, isomeric:bool=False, kekulize:bool=False,
                 min_radius:int=1, length:int=2048, n_permutations:int=0, seed:int=0, parallel: Union[bool, int] = False,):
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
        super().__init__(parallel = parallel)
        self.radius = radius
        self.rings = rings
        self.isomeric = isomeric
        self.kekulize = kekulize
        self.min_radius = min_radius
        self.length = length
        #Set the .n_permutations and seed without creating the encoder twice
        self._n_permutations = n_permutations
        self._seed = seed
        # create the encoder instance
        self._recreate_encoder()

    def __getstate__(self):
        # Get the state of the parent class
        state = super().__getstate__()
        # Remove the unpicklable property from the state
        state.pop("mhfp_encoder", None) # mhfp_encoder is not picklable
        return state

    def __setstate__(self, state):
        # Restore the state of the parent class
        super().__setstate__(state)
        # Re-create the unpicklable property
        self._recreate_encoder()

    def _mol2fp(self, mol):
        return self.mhfp_encoder.EncodeSECFPMol(mol, self.radius, self.rings, self.isomeric, self.kekulize, self.min_radius, self.length) 

    def _recreate_encoder(self):
        self.mhfp_encoder = rdMHFPFingerprint.MHFPEncoder(self._n_permutations, self._seed)

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

class MorganFingerprintTransformer(FpsTransformer):
    def __init__(self, nBits=2048, radius=2, useChirality=False, useBondTypes=True, useFeatures=False, useCounts=False, parallel: Union[bool, int] = False,):
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
        super().__init__(parallel = parallel)
        self.nBits = nBits
        self.radius = radius
        self.useChirality = useChirality
        self.useBondTypes = useBondTypes
        self.useFeatures = useFeatures
        self.useCounts = useCounts

    def _mol2fp(self, mol):
        if self.useCounts:
            return rdMolDescriptors.GetHashedMorganFingerprint(
                mol,int(self.radius),nBits=int(self.nBits), useFeatures=bool(self.useFeatures),
                useChirality=bool(self.useChirality), useBondTypes=bool(self.useBondTypes)
            )
        else:
            return rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol,int(self.radius),nBits=int(self.nBits), useFeatures=bool(self.useFeatures),
                useChirality=bool(self.useChirality), useBondTypes=bool(self.useBondTypes)
            )
        
class AvalonFingerprintTransformer(FpsTransformer):
    # Fingerprint from the Avalon toolkeit, https://doi.org/10.1021/ci050413p
    def __init__(self, nBits:int = 512, isQuery:bool = False, resetVect:bool = False, bitFlags:int = 15761407, useCounts:bool = False, parallel: Union[bool, int] = False,):
        """ Transform RDKit mols into Count or bit-based Avalon Fingerprints

        Parameters
        ----------
        nBits : int, optional
            Size of the fingerprint, by default 512
        isQuery : bool, optional
            use the fingerprint for a query structure, by default False
        resetVect : bool, optional
            reset vector, by default False      NB: only used in GetAvalonFP (not for GetAvalonCountFP)
        bitFlags : int, optional
            Substructure fingerprint (32767) or similarity fingerprint (15761407) by default 15761407
        useCounts : bool, optional
            If toggled will create the count and not bit-based fingerprint, by default False
        """
        super().__init__(parallel = parallel)
        self.nBits = nBits
        self.isQuery = isQuery
        self.resetVect = resetVect
        self.bitFlags = bitFlags
        self.useCounts = useCounts
        
    def _mol2fp(self, mol):
        if self.useCounts:
            return pyAvalonTools.GetAvalonCountFP(mol,
                                                  nBits=int(self.nBits),
                                                  isQuery=bool(self.isQuery),
                                                  bitFlags=int(self.bitFlags)
            )
        else:
            return pyAvalonTools.GetAvalonFP(mol,
                                             nBits=int(self.nBits),
                                             isQuery=bool(self.isQuery),
                                             resetVect=bool(self.resetVect),
                                             bitFlags=int(self.bitFlags)                      
            )


def parallel_helper(args):
    """Parallel_helper takes a tuple with classname, the objects parameters and the mols to process.
    Then instantiates the class with the parameters and processes the mol.
    Intention is to be able to do this in chilcprocesses as some classes can't be pickled"""
    classname, parameters, X_mols = args
    from scikit_mol import fingerprints
    transformer = getattr(fingerprints, classname)(**parameters)
    return transformer._transform(X_mols)

