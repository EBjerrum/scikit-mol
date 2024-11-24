from typing import Union
from rdkit.Chem import rdMolDescriptors
import numpy as np

from .baseclasses import FpsTransformer


class MACCSKeysFingerprintTransformer(FpsTransformer):
    def __init__(
        self,
        parallel: Union[bool, int] = False,
        safe_inference_mode: bool = False,
        dtype: np.dtype = np.int8,
        fpSize=167,
    ):
        """MACCS keys fingerprinter
        calculates the 167 fixed MACCS keys
        """
        super().__init__(
            parallel=parallel, safe_inference_mode=safe_inference_mode, dtype=dtype
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
