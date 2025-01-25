from typing import Optional

import numpy as np
from rdkit.Avalon import pyAvalonTools

from .baseclasses import FpsTransformer


class AvalonFingerprintTransformer(FpsTransformer):
    # Fingerprint from the Avalon toolkeit, https://doi.org/10.1021/ci050413p
    def __init__(
        self,
        fpSize: int = 512,
        isQuery: bool = False,
        resetVect: bool = False,
        bitFlags: int = 15761407,
        useCounts: bool = False,
        n_jobs: Optional[int] = None,
        safe_inference_mode: bool = False,
        dtype: np.dtype = np.int8,
    ):
        """Transform RDKit mols into Count or bit-based Avalon Fingerprints

        Parameters
        ----------
        fpSize : int, optional
            Size of the fingerprint, by default 512
        isQuery : bool, optional
            use the fingerprint for a query structure, by default False
        resetVect : bool, optional
            reset vector, by default False      NB: only used in GetAvalonFP (not for GetAvalonCountFP)
        bitFlags : int, optional
            Substructure fingerprint (32767) or similarity fingerprint (15761407) by default 15761407
        useCounts : bool, optional
            If toggled will create the count and not bit-based fingerprint, by default False
        """
        super().__init__(
            n_jobs=n_jobs, safe_inference_mode=safe_inference_mode, dtype=dtype
        )
        self.fpSize = fpSize
        self.isQuery = isQuery
        self.resetVect = resetVect
        self.bitFlags = bitFlags
        self.useCounts = useCounts

    def _mol2fp(self, mol):
        if self.useCounts:
            return pyAvalonTools.GetAvalonCountFP(
                mol,
                nBits=int(self.fpSize),
                isQuery=bool(self.isQuery),
                bitFlags=int(self.bitFlags),
            )
        else:
            return pyAvalonTools.GetAvalonFP(
                mol,
                nBits=int(self.fpSize),
                isQuery=bool(self.isQuery),
                resetVect=bool(self.resetVect),
                bitFlags=int(self.bitFlags),
            )
