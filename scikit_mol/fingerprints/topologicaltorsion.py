from typing import Optional, Sequence

import numpy as np
from rdkit.Chem.rdFingerprintGenerator import GetTopologicalTorsionGenerator

from .baseclasses import FpsGeneratorTransformer


class TopologicalTorsionFingerprintTransformer(FpsGeneratorTransformer):
    """
    Transformer for generating topological torsion fingerprints.
    """

    _regenerate_on_properties = ("fpSize", "includeChirality", "targetSize")

    def __init__(
        self,
        targetSize: int = 4,
        fromAtoms: Optional[Sequence] = None,
        ignoreAtoms: Optional[Sequence] = None,
        atomInvariants: Optional[Sequence] = None,
        confId: int = -1,
        includeChirality: bool = False,
        fpSize: int = 2048,
        useCounts: bool = False,
        n_jobs: Optional[int] = None,
        safe_inference_mode: bool = False,
    ):
        """
        Parameters
        ----------
        targetSize : int, optional
            The number of atoms to include in the torsion, by default 4.
        fromAtoms : list, optional
            Atom indices to include in the fingerprint generation, by default None.
        ignoreAtoms : list, optional
            Atom indices to exclude in the fingerprint generation, by default None.
        atomInvariants : list, optional
            Custom atom invariants to be used, by default None.
        confId : int, optional
            Conformation ID to use, by default -1.
        includeChirality : bool, optional
            Whether to include chirality in the fingerprint, by default False.
        fpSize : int, optional
            Size of the fingerprint, by default 2048.
        useCounts : bool, optional
            Whether to use counts in the fingerprint, by default False.
        n_jobs : int, optional default=None
            The maximum number of concurrently running jobs.
            None is a marker for 'unset' that will be interpreted as `n_jobs=1` unless the call is performed under a `joblib.parallel_config()` context manager that sets another value for `n_jobs`.
        safe_inference_mode : bool, optional
            If `True`, will return masked arrays for invalid mols, by default `False`
        """
        self._initializing = True
        super().__init__(n_jobs=n_jobs, safe_inference_mode=safe_inference_mode)
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
        """
        Generate the fingerprint generator.
        """
        self._fpgen = GetTopologicalTorsionGenerator(
            torsionAtomCount=int(self.targetSize),
            includeChirality=bool(self.includeChirality),
            fpSize=int(self.fpSize),
        )

    def _transform_mol(self, mol) -> np.array:
        """
        Transform a molecule into its fingerprint representation.

        Parameters
        ----------
        mol : RDKit Mol
            The molecule to transform.

        Returns
        -------
        np.array
            The fingerprint of the molecule as a NumPy array.
        """
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
