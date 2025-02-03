from typing import Optional

import numpy as np
from rdkit.Chem import rdMolDescriptors

from .baseclasses import FpsTransformer


class MACCSKeysFingerprintTransformer(FpsTransformer):
    def __init__(
        self,
        n_jobs: Optional[int] = None,
        safe_inference_mode: bool = False,
        dtype: np.dtype = np.int8,
        fpSize=167,
    ):
        """MACCS keys fingerprinter
        calculates the 167 fixed MACCS keys
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
