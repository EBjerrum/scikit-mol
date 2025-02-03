from typing import Optional

import numpy as np
from rdkit.Chem.rdFingerprintGenerator import GetRDKitFPGenerator

from .baseclasses import FpsGeneratorTransformer


class RDKitFingerprintTransformer(FpsGeneratorTransformer):
    _regenerate_on_properties = (
        "minPath",
        "maxPath",
        "useHs",
        "branchedPaths",
        "useBondOrder",
        "countSimulation",
        "fpSize",
        "countBounds",
        "numBitsPerFeature",
    )

    def __init__(
        self,
        minPath: int = 1,
        maxPath: int = 7,
        useHs: bool = True,
        branchedPaths: bool = True,
        useBondOrder: bool = True,
        countSimulation: bool = False,
        countBounds=None,
        fpSize: int = 2048,
        numBitsPerFeature: int = 2,
        useCounts: bool = False,
        n_jobs: Optional[int] = None,
        safe_inference_mode: bool = False,
    ):
        """Calculates the RDKit fingerprints

        Parameters
        ----------
        minPath : int, optional
            the minimum path length (in bonds) to be included, by default 1
        maxPath : int, optional
            the maximum path length (in bonds) to be included, by default 7
        useHs : bool, optional
            toggles inclusion of Hs in paths (if the molecule has explicit Hs), by default True
        branchedPaths : bool, optional
            toggles generation of branched subgraphs, not just linear paths, by default True
        useBondOrder : bool, optional
            toggles inclusion of bond orders in the path hashes, by default True
        countSimulation : bool, optional
            if set, use count simulation while generating the fingerprint, by default False
        countBounds : _type_, optional
            boundaries for count simulation, corresponding bit will be set if the count is higher than the number provided for that spot, by default None
        fpSize : int, optional
            size of the generated fingerprint, does not affect the sparse versions, by default 2048
        numBitsPerFeature : int, optional
            the number of bits set per path/subgraph found, by default 2
        """
        self._initializing = True
        super().__init__(n_jobs=n_jobs, safe_inference_mode=safe_inference_mode)
        self.minPath = minPath
        self.maxPath = maxPath
        self.useHs = useHs
        self.branchedPaths = branchedPaths
        self.useBondOrder = useBondOrder
        self.countSimulation = countSimulation
        self.fpSize = fpSize
        self.numBitsPerFeature = numBitsPerFeature
        self.countBounds = countBounds

        self.useCounts = useCounts

        self._generate_fp_generator()
        delattr(self, "_initializing")

    def _transform_mol(self, mol) -> np.array:
        if self.useCounts:
            return self._fpgen.GetCountFingerprintAsNumPy(mol)
        else:
            return self._fpgen.GetFingerprintAsNumPy(mol)

    def _generate_fp_generator(self):
        self._fpgen = GetRDKitFPGenerator(
            minPath=int(self.minPath),
            maxPath=int(self.maxPath),
            useHs=bool(self.useHs),
            branchedPaths=bool(self.branchedPaths),
            useBondOrder=bool(self.useBondOrder),
            countSimulation=bool(self.countSimulation),
            fpSize=int(self.fpSize),
            countBounds=bool(self.countBounds),
            numBitsPerFeature=int(self.numBitsPerFeature),
        )
