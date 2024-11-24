from typing import Union

from rdkit.Chem import rdMolDescriptors

import numpy as np

from warnings import warn

from rdkit.Chem.rdFingerprintGenerator import (
    GetMorganGenerator,
    GetMorganFeatureAtomInvGen,
)

from .baseclasses import FpsTransformer, FpsGeneratorTransformer


class MorganFingerprintTransformerClassic(FpsTransformer):
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
        dtype: np.dtype = np.int8,
    ):
        """Transform RDKit mols into Count or bit-based hashed MorganFingerprints

        Parameters
        ----------
        fpSize : int, optional
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
        super().__init__(
            parallel=parallel, safe_inference_mode=safe_inference_mode, dtype=dtype
        )
        self.fpSize = fpSize
        self.radius = radius
        self.useChirality = useChirality
        self.useBondTypes = useBondTypes
        self.useFeatures = useFeatures
        self.useCounts = useCounts

        warn(
            "MorganFingerprintTransformer will be replace by MorganGeneratorTransformer, due to changes in RDKit!",
            DeprecationWarning,
        )

    def _mol2fp(self, mol):
        if self.useCounts:
            return rdMolDescriptors.GetHashedMorganFingerprint(
                mol,
                int(self.radius),
                nBits=int(self.fpSize),
                useFeatures=bool(self.useFeatures),
                useChirality=bool(self.useChirality),
                useBondTypes=bool(self.useBondTypes),
            )
        else:
            return rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol,
                int(self.radius),
                nBits=int(self.fpSize),
                useFeatures=bool(self.useFeatures),
                useChirality=bool(self.useChirality),
                useBondTypes=bool(self.useBondTypes),
            )


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
        nBits: int = None,
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


class MorganFPGeneratorTransformer(FpsGeneratorTransformer):
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
        nBits: int = None,
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
