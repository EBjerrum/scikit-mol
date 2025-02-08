from typing import Optional, Sequence

import numpy as np
from rdkit.Chem.rdFingerprintGenerator import GetAtomPairGenerator

from .baseclasses import FpsGeneratorTransformer


class AtomPairFingerprintTransformer(FpsGeneratorTransformer):
    """
    AtomPair fingerprints encode pairs of atoms at various topological or 3D distances in a molecule.
    They are useful for capturing structural relationships and connectivity patterns.

    """

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
        fromAtoms: Optional[Sequence] = None,
        ignoreAtoms: Optional[Sequence] = None,
        atomInvariants: Optional[Sequence] = None,
        includeChirality: bool = False,
        use2D: bool = True,
        confId: int = -1,
        fpSize: int = 2048,
        useCounts: bool = False,
        n_jobs: Optional[int] = None,
        safe_inference_mode: bool = False,
    ):
        """Transform RDKit mols into Count or bit-based hashed AtomPair Fingerprints

        Parameters
        ----------
        minLength : int, optional
            Minimum distance between atom pairs, by default 1
        maxLength : int, optional
            Maximum distance between atom pairs, by default 30
        fromAtoms : Sequence, optional
            Atom indices to use as starting points, by default None
        ignoreAtoms : array-like, optional
            Atom indices to exclude, by default None
        atomInvariants : array-like, optional
            Atom invariants to use, by default None
        includeChirality : bool, optional
            Include chirality in calculation of the fingerprint keys, by default False
        use2D : bool, optional
            Use 2D distances (topological) instead of 3D, by default True
        confId : int, optional
            Which conformer to use for 3D distance calculations, by default -1
        fpSize : int, optional
            Size of the hashed fingerprint, by default 2048
        useCounts : bool, optional
            If toggled will create the count and not bit-based fingerprint, by default False
        n_jobs : int, optional
            The maximum number of concurrently running jobs.
            `None` is a marker for 'unset' that will be interpreted as `n_jobs=1` unless the call is performed under a
            `joblib.parallel_config()` context manager that sets another value for `n_jobs`.
        safe_inference_mode : bool, optional
            If `True`, will return masked arrays for invalid mols, by default `False`
        """
        self._initializing = True
        super().__init__(n_jobs=n_jobs, safe_inference_mode=safe_inference_mode)
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
