from typing import Union

import numpy as np

from warnings import warn

from .baseclasses import FpsTransformer, FpsGeneratorTransformer

from rdkit.Chem.rdFingerprintGenerator import GetAtomPairGenerator
from rdkit.Chem import rdMolDescriptors


class AtomPairFingerprintTransformer(FpsGeneratorTransformer):
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
            minDistance=int(self.minLength),
            maxDistance=int(self.maxLength),
            includeChirality=bool(self.includeChirality),
            use2D=bool(self.use2D),
            fpSize=int(self.fpSize),
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
