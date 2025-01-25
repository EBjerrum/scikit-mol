from typing import Optional
from warnings import warn

import numpy as np
from rdkit.Chem import rdMHFPFingerprint

from .baseclasses import FpsTransformer


# TODO move to use FpsGeneratorTransformer
class MHFingerprintTransformer(FpsTransformer):
    def __init__(
        self,
        radius: int = 3,
        rings: bool = True,
        isomeric: bool = False,
        kekulize: bool = False,
        min_radius: int = 1,
        fpSize: int = 2048,
        seed: int = 42,
        n_jobs: Optional[int] = None,
        safe_inference_mode: bool = False,
        dtype: np.dtype = np.int32,
    ):
        """Transforms the RDKit mol into the MinHash fingerprint (MHFP)

        https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0321-8

        Args:
            radius (int, optional): The MHFP radius. Defaults to 3.
            rings (bool, optional): Whether or not to include rings in the shingling. Defaults to True.
            isomeric (bool, optional): Whether the isomeric SMILES to be considered. Defaults to False.
            kekulize (bool, optional): Whether or not to kekulize the extracted SMILES. Defaults to False.
            min_radius (int, optional): The minimum radius that is used to extract n-gram. Defaults to 1.
            fpSize (int, optional): The number of permutations used for hashing. Defaults to 2048,
            this is effectively the length of the FP
            seed (int, optional): The value used to seed numpy.random. Defaults to 0.
        """
        super().__init__(
            n_jobs=n_jobs, safe_inference_mode=safe_inference_mode, dtype=dtype
        )
        self.radius = radius
        self.rings = rings
        self.isomeric = isomeric
        self.kekulize = kekulize
        self.min_radius = min_radius
        # Set the .n_permutations and .seed without creating the encoder twice
        self.fpSize = fpSize
        self._seed = seed
        # create the encoder instance
        self._recreate_encoder()

    def __getstate__(self):
        # Get the state of the parent class
        state = super().__getstate__()
        # Remove the unpicklable property from the state
        state.pop("mhfp_encoder", None)  # mhfp_encoder is not picklable
        return state

    def __setstate__(self, state):
        # Restore the state of the parent class
        super().__setstate__(state)
        # Re-create the unpicklable property
        self._recreate_encoder()

    def _mol2fp(self, mol):
        fp = self.mhfp_encoder.EncodeMol(
            mol, self.radius, self.rings, self.isomeric, self.kekulize, self.min_radius
        )
        return fp

    def _fp2array(self, fp):
        return np.array(fp)

    def _recreate_encoder(self):
        self.mhfp_encoder = rdMHFPFingerprint.MHFPEncoder(
            int(self.fpSize), int(self._seed)
        )

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
        warn(
            "n_permutations will be replace by fpSize, due to changes harmonization!",
            DeprecationWarning,
        )
        return self.fpSize

    @n_permutations.setter
    def n_permutations(self, n_permutations):
        warn(
            "n_permutations will be replace by fpSize, due to changes harmonization!",
            DeprecationWarning,
        )
        self.fpSize = n_permutations
        # each time the n_permutations parameter is modified refresh an instance of the encoder
        self._recreate_encoder()


# TODO use FpsGeneratorTransformer instead
class SECFingerprintTransformer(FpsTransformer):
    # https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0321-8
    def __init__(
        self,
        radius: int = 3,
        rings: bool = True,
        isomeric: bool = False,
        kekulize: bool = False,
        min_radius: int = 1,
        fpSize: int = 2048,
        n_permutations: int = 0,
        seed: int = 0,
        n_jobs: Optional[int] = None,
        safe_inference_mode: bool = False,
        dtype: np.dtype = np.int8,
    ):
        """Transforms the RDKit mol into the SMILES extended connectivity fingerprint (SECFP)

        Args:
            radius (int, optional): The MHFP radius. Defaults to 3.
            rings (bool, optional): Whether or not to include rings in the shingling. Defaults to True.
            isomeric (bool, optional): Whether the isomeric SMILES to be considered. Defaults to False.
            kekulize (bool, optional): Whether or not to kekulize the extracted SMILES. Defaults to False.
            min_radius (int, optional): The minimum radius that is used to extract n-gram. Defaults to 1.
            fpSize (int, optional): The length of the folded fingerprint. Defaults to 2048.
            n_permutations (int, optional): The number of permutations used for hashing. Defaults to 0.
            seed (int, optional): The value used to seed numpy.random. Defaults to 0.
        """
        super().__init__(
            n_jobs=n_jobs, safe_inference_mode=safe_inference_mode, dtype=dtype
        )
        self.radius = radius
        self.rings = rings
        self.isomeric = isomeric
        self.kekulize = kekulize
        self.min_radius = min_radius
        self.fpSize = fpSize
        # Set the .n_permutations and seed without creating the encoder twice
        self._n_permutations = n_permutations
        self._seed = seed
        # create the encoder instance
        self._recreate_encoder()

    def __getstate__(self):
        # Get the state of the parent class
        state = super().__getstate__()
        # Remove the unpicklable property from the state
        state.pop("mhfp_encoder", None)  # mhfp_encoder is not picklable
        return state

    def __setstate__(self, state):
        # Restore the state of the parent class
        super().__setstate__(state)
        # Re-create the unpicklable property
        self._recreate_encoder()

    def _mol2fp(self, mol):
        return self.mhfp_encoder.EncodeSECFPMol(
            mol,
            int(self.radius),
            bool(self.rings),
            bool(self.isomeric),
            bool(self.kekulize),
            int(self.min_radius),
            int(self.fpSize),
        )

    def _recreate_encoder(self):
        self.mhfp_encoder = rdMHFPFingerprint.MHFPEncoder(
            self._n_permutations, self._seed
        )

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
    def length(self):
        warn(
            "length will be replace by fpSize, due to changes harmonization!",
            DeprecationWarning,
        )
        return self.fpSize
