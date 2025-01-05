from typing import Optional, Union

import numpy as np
from rdkit.Chem.rdFingerprintGenerator import (
    GetMorganFeatureAtomInvGen,
    GetMorganGenerator,
)

from .baseclasses import FpsGeneratorTransformer


class MorganFingerprintTransformer(FpsGeneratorTransformer):
    _regenerate_on_properties = (
        "radius",
        "fpSize",
        "useChirality",
        "useFeatures",
        "useBondTypes",
    )

    def __init__(
        self,
        fpSize=2048,
        radius=2,
        useChirality=False,
        useBondTypes=True,
        useFeatures=False,
        useCounts=False,
        parallel: Union[bool, int] = False,
        safe_inference_mode: bool = False,
        dtype: np.dtype = None,
        nBits: Optional[int] = None,
    ):
        """Transform RDKit mols into Count or bit-based hashed MorganFingerprints

        Parameters
        ----------
        fpsize : int, optional
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
        parallel : bool or int, optional
            If True, will use all available cores, if int will use that many cores, by default False
        safe_inference_mode : bool, optional
            If True, will return masked arrays for invalid mols, by default False
        """

        self._initializing = True
        super().__init__(parallel=parallel, safe_inference_mode=safe_inference_mode)
        self.fpSize = fpSize
        self.radius = radius
        self.useChirality = useChirality
        self.useFeatures = useFeatures
        self.useCounts = useCounts
        self.useBondTypes = useBondTypes
        self.dtype = dtype
        self.nBits = nBits

        self._generate_fp_generator()
        delattr(self, "_initializing")

    def _generate_fp_generator(self):
        if self.useFeatures:
            atomInvariantsGenerator = GetMorganFeatureAtomInvGen()
        else:
            atomInvariantsGenerator = None

        self._fpgen = GetMorganGenerator(
            radius=int(self.radius),
            fpSize=int(self.fpSize),
            includeChirality=bool(self.useChirality),
            useBondTypes=bool(self.useBondTypes),
            atomInvariantsGenerator=atomInvariantsGenerator,
        )

    def _transform_mol(self, mol) -> np.array:
        if self.useCounts:
            return self._fpgen.GetCountFingerprintAsNumPy(mol)
        else:
            return self._fpgen.GetFingerprintAsNumPy(mol)
