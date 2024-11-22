from typing import Union

import numpy as np

from warnings import warn

from .baseclasses import FpsTransformer, FpsGeneratorTransformer

from rdkit.Chem.rdFingerprintGenerator import GetAtomPairGenerator
from rdkit.Chem import rdMolDescriptors


class AtomPairFingerprintTransformer(FpsTransformer):
    def __init__(
        self,
        minLength: int = 1,
        maxLength: int = 30,
        fromAtoms=0,
        ignoreAtoms=0,
        atomInvariants=0,
        nBitsPerEntry: int = 4,
        includeChirality: bool = False,
        use2D: bool = True,
        confId: int = -1,
        fpSize=2048,
        useCounts: bool = False,
        parallel: Union[bool, int] = False,
        safe_inference_mode: bool = False,
        dtype: np.dtype = np.int8,
    ):
        super().__init__(
            parallel=parallel, safe_inference_mode=safe_inference_mode, dtype=dtype
        )
        self.minLength = minLength
        self.maxLength = maxLength
        self.fromAtoms = fromAtoms
        self.ignoreAtoms = ignoreAtoms
        self.atomInvariants = atomInvariants
        self.includeChirality = includeChirality
        self.use2D = use2D
        self.confId = confId
        self.fpSize = fpSize
        self.nBitsPerEntry = nBitsPerEntry
        self.useCounts = useCounts

        warn(
            "AtomPairFingerprintTransformer will be replace by AtomPairFPGeneratorTransformer, due to changes in RDKit!",
            DeprecationWarning,
        )

    def _mol2fp(self, mol):
        if self.useCounts:
            return rdMolDescriptors.GetHashedAtomPairFingerprint(
                mol,
                nBits=int(self.fpSize),
                minLength=int(self.minLength),
                maxLength=int(self.maxLength),
                fromAtoms=self.fromAtoms,
                ignoreAtoms=self.ignoreAtoms,
                atomInvariants=self.atomInvariants,
                includeChirality=bool(self.includeChirality),
                use2D=bool(self.use2D),
                confId=int(self.confId),
            )
        else:
            return rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
                mol,
                nBits=int(self.fpSize),
                minLength=int(self.minLength),
                maxLength=int(self.maxLength),
                fromAtoms=self.fromAtoms,
                ignoreAtoms=self.ignoreAtoms,
                atomInvariants=self.atomInvariants,
                nBitsPerEntry=int(self.nBitsPerEntry),
                includeChirality=bool(self.includeChirality),
                use2D=bool(self.use2D),
                confId=int(self.confId),
            )


class AtomPairFPGeneratorTransformer(FpsGeneratorTransformer):
    _regenerate_on_properties = (
        "fpSize",
        "includeChirality",
        "use2D",
        "minLength",
        "maxLength",
    )

    def __init__(
        self,
        minLength: int = 1,
        maxLength: int = 30,
        fromAtoms=None,
        ignoreAtoms=None,
        atomInvariants=None,
        includeChirality: bool = False,
        use2D: bool = True,
        confId: int = -1,
        fpSize: int = 2048,
        useCounts: bool = False,
        parallel: Union[bool, int] = False,
        safe_inference_mode: bool = False,
    ):
        self._initializing = True
        super().__init__(parallel=parallel, safe_inference_mode=safe_inference_mode)
        self.fpSize = fpSize
        self.use2D = use2D
        self.includeChirality = includeChirality
        self.minLength = minLength
        self.maxLength = maxLength

        self.useCounts = useCounts
        self.confId = confId
        self.fromAtoms = fromAtoms
        self.ignoreAtoms = ignoreAtoms
        self.atomInvariants = atomInvariants
        self._generate_fp_generator()
        delattr(self, "_initializing")

    def _generate_fp_generator(self):
        self._fpgen = GetAtomPairGenerator(
            minDistance=self.minLength,
            maxDistance=self.maxLength,
            includeChirality=self.includeChirality,
            use2D=self.use2D,
            fpSize=self.fpSize,
        )

    def _transform_mol(self, mol) -> np.array:
        if self.useCounts:
            return self._fpgen.GetCountFingerprintAsNumPy(
                mol,
                fromAtoms=self.fromAtoms,
                ignoreAtoms=self.ignoreAtoms,
                customAtomInvariants=self.atomInvariants,
            )
        else:
            return self._fpgen.GetFingerprintAsNumPy(
                mol,
                fromAtoms=self.fromAtoms,
                ignoreAtoms=self.ignoreAtoms,
                customAtomInvariants=self.atomInvariants,
            )
