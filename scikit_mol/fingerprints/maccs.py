from typing import Optional

import numpy as np
from rdkit.Chem import rdMolDescriptors

from .baseclasses import FpsTransformer


class MACCSKeysFingerprintTransformer(FpsTransformer):
    """MACCS keys fingerprinter calculates the 167 fixed MACCS keys"""

    def __init__(
        self,
        n_jobs: Optional[int] = None,
        safe_inference_mode: bool = False,
        dtype: np.dtype = np.int8,
        fpSize=167,
    ):
        """

        Parameters
        ----------
        n_jobs : int, optional default=None
            The maximum number of concurrently running jobs.
            None is a marker for 'unset' that will be interpreted as `n_jobs=1` unless the call is performed under a `joblib.parallel_config()` context manager that sets another value for `n_jobs`.
        safe_inference_mode : bool, optional
            If `True`, will return masked arrays for invalid mols, by default `False`
        dtype : np.dtype, optional
            Data type of the fingerprint array, by default np.int8
        fpSize : int, optional
            Size of the fingerprint, by default 167

        Raises
        ------
        ValueError
            _description_
        """

        super().__init__(
            n_jobs=n_jobs, safe_inference_mode=safe_inference_mode, dtype=dtype
        )
        if fpSize != 167:
            raise ValueError(
                "fpSize can only be 167, matching the number of defined MACCS keys!"
            )
        self._fpSize = fpSize

    @property
    def fpSize(self):
        return self._fpSize

    @fpSize.setter
    def fpSize(self, fpSize):
        if fpSize != 167:
            raise ValueError(
                "fpSize can only be 167, matching the number of defined MACCS keys!"
            )
        self._fpSize = fpSize

    def _mol2fp(self, mol):
        return rdMolDescriptors.GetMACCSKeysFingerprint(mol)
