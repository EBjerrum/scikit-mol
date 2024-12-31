from typing import Union

import numpy as np

from warnings import warn

from .baseclasses import FpsTransformer, FpsGeneratorTransformer

from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetTopologicalTorsionGenerator


class TopologicalTorsionFingerprintTransformer(FpsGeneratorTransformer):
    _regenerate_on_properties = ("fpSize", "includeChirality", "targetSize")

    def __init__(
        self,
        targetSize: int = 4,
        fromAtoms=None,
        ignoreAtoms=None,
        atomInvariants=None,
        confId=-1,
        includeChirality: bool = False,
        fpSize: int = 2048,
        useCounts: bool = False,
        parallel: Union[bool, int] = False,
        safe_inference_mode: bool = False,
    ):
        self._initializing = True
        super().__init__(parallel=parallel, safe_inference_mode=safe_inference_mode)
        self.fpSize = fpSize
        self.includeChirality = includeChirality
        self.targetSize = targetSize

        self.fromAtoms = fromAtoms
        self.ignoreAtoms = ignoreAtoms
        self.atomInvariants = atomInvariants
        self.confId = confId
        self.useCounts = useCounts

        self._generate_fp_generator()
        delattr(self, "_initializing")

    def _generate_fp_generator(self):
        self._fpgen = GetTopologicalTorsionGenerator(
            torsionAtomCount=int(self.targetSize),
            includeChirality=bool(self.includeChirality),
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
