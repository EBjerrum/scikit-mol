from multiprocessing import Pool, get_context
import multiprocessing
import re
import inspect
from warnings import warn, simplefilter

from typing import Union
from rdkit import DataStructs

# from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdMHFPFingerprint


from rdkit.Chem.rdFingerprintGenerator import (
    GetMorganGenerator,
    GetMorganFeatureAtomInvGen,
    GetTopologicalTorsionGenerator,
    GetAtomPairGenerator,
    GetRDKitFPGenerator,
)

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from scipy.sparse import vstack

from sklearn.base import BaseEstimator, TransformerMixin
from scikit_mol.core import check_transform_input

from abc import ABC, abstractmethod

simplefilter("always", DeprecationWarning)

_PATTERN_FINGERPRINT_TRANSFORMER = re.compile(
    r"^(?P<fingerprint_name>\w+)FingerprintTransformer$"
)


class BaseFpsTransformer(ABC, BaseEstimator, TransformerMixin):
    def __init__(
        self,
        parallel: Union[bool, int] = False,
        start_method: str = None,
        safe_inference_mode: bool = False,
    ):
        self.parallel = parallel
        self.start_method = start_method
        self.safe_inference_mode = safe_inference_mode

    # TODO, remove when finally deprecating nBits and dtype
    @property
    def nBits(self):
        warn(
            "nBits will be replaced by fpSize, due to changes harmonization!",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.fpSize

    # TODO, remove when finally deprecating nBits and dtype
    @nBits.setter
    def nBits(self, nBits):
        if nBits is not None:
            warn(
                "nBits will be replaced by fpSize, due to changes harmonization!",
                DeprecationWarning,
                stacklevel=3,
            )
            self.fpSize = nBits

    def _get_column_prefix(self) -> str:
        matched = _PATTERN_FINGERPRINT_TRANSFORMER.match(type(self).__name__)
        if matched:
            fingerprint_name = matched.group("fingerprint_name")
            return f"fp_{fingerprint_name.lower()}"
        else:
            return "fp"

    def _get_n_digits_column_suffix(self) -> int:
        return len(str(self.fpSize))

    def get_display_feature_names_out(self, input_features=None):
        """Get feature names for display purposes

        All feature names will have the same length,
        since the different elements will be prefixed with zeros
        depending on the number of bits.
        """
        prefix = self._get_column_prefix()
        n_digits = self._get_n_digits_column_suffix()
        return np.array(
            [f"{prefix}_{str(i).zfill(n_digits)}" for i in range(1, self.fpSize + 1)]
        )

    def get_feature_names_out(self, input_features=None):
        """Get feature names for fingerprint transformers

        This method is used by the scikit-learn set_output API
        to get the column names of the transformed dataframe.
        """
        prefix = self._get_column_prefix()
        return np.array([f"{prefix}_{i}" for i in range(1, self.fpSize + 1)])

    def _safe_transform_mol(self, mol):
        """Handle safe inference mode with masked arrays"""
        if not mol and self.safe_inference_mode:
            return np.ma.masked_all(self.fpSize)

        try:
            result = self._transform_mol(mol)
            return result
        except Exception as e:
            if self.safe_inference_mode:
                return np.ma.masked_all(self.fpSize)
            else:
                raise e

    @abstractmethod
    def _transform_mol(self, mol):
        """Transform a single molecule to numpy array"""
        raise NotImplementedError

    def fit(self, X, y=None):
        """Included for scikit-learn compatibility

        Also sets the column prefix for use by the transform method with dataframe output.
        """
        return self

    @check_transform_input
    def _transform(self, X):
        if self.safe_inference_mode:
            # Use the new method with masked arrays if we're in safe inference mode
            arrays = [self._safe_transform_mol(mol) for mol in X]
            return np.ma.stack(arrays)
        elif hasattr(
            self, "dtype"
        ):  # TODO, it seems a bit of a code smell that we have to preemptively test a property from the baseclass?
            # Use the original, faster method if we're not in safe inference mode
            # This also triggers a deprecation warning!
            arr = np.zeros((len(X), self.fpSize), dtype=self.dtype)
            for i, mol in enumerate(X):
                arr[i, :] = self._transform_mol(mol)
            return arr
        else:  # We are unsure on the dtype, so we don't use a preassigned array #TODO test time differnece to previous
            arrays = [self._transform_mol(mol) for mol in X]
            return np.stack(arrays)

    def _transform_sparse(self, X):
        arr = np.zeros((len(X), self.fpSize), dtype=self.dtype)
        for i, mol in enumerate(X):
            arr[i, :] = self._transform_mol(mol)

        return lil_matrix(arr)

    def transform(self, X, y=None):
        """Transform a list of RDKit molecule objects into a fingerprint array

        Parameters
        ----------
        X : (List, np.array, pd.Series)
            A list of RDKit molecules
        y : NoneType, optional
            Target values for scikit-learn compatibility, not used, by default None

        Returns
        -------
        np.array
            Fingerprints, shape (samples, fingerprint size)
        """
        if not self.parallel:
            return self._transform(X)

        elif self.parallel:
            n_processes = (
                self.parallel if self.parallel > 1 else None
            )  # Pool(processes=None) autodetects
            n_chunks = (
                n_processes if n_processes is not None else multiprocessing.cpu_count()
            )

            with get_context(self.start_method).Pool(processes=n_processes) as pool:
                x_chunks = np.array_split(X, n_chunks)
                # TODO check what is fastest, pickle or recreate and do this only for classes that need this
                # arrays = pool.map(self._transform, x_chunks)
                parameters = self.get_params()
                # TODO: create "transform_parallel" function in the core module,
                # and use it here and in the descriptors transformer
                # x_chunks = [np.array(x).reshape(-1, 1) for x in x_chunks]
                arrays = pool.map(
                    parallel_helper,
                    [
                        (self.__class__.__name__, parameters, x_chunk)
                        for x_chunk in x_chunks
                    ],
                )
                if self.safe_inference_mode:
                    arr = np.ma.concatenate(arrays)
                else:
                    arr = np.concatenate(arrays)
            return arr


class FpsTransformer(BaseFpsTransformer):
    """Classic fingerprint transformer using mol2fp pattern"""

    def __init__(
        self,
        parallel: Union[bool, int] = False,
        safe_inference_mode: bool = False,
        dtype: np.dtype = np.int8,
    ):
        super().__init__(parallel=parallel, safe_inference_mode=safe_inference_mode)
        self.dtype = dtype

    def _transform_mol(self, mol):
        """Implements the mol -> rdkit fingerprint data structure -> numpy array pattern"""
        fp = self._mol2fp(mol)
        return self._fp2array(fp)

    @abstractmethod
    def _mol2fp(self, mol):
        """Generate fingerprint from mol

        MUST BE OVERWRITTEN
        """
        raise NotImplementedError("_mol2fp not implemented")

    def _fp2array(self, fp):
        """Convert RDKit fingerprint data structure to numpy array"""
        if fp:
            arr = np.zeros((self.fpSize,), dtype=self.dtype)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        else:
            return np.ma.masked_all((self.fpSize,), dtype=self.dtype)

    # TODO, remove when finally deprecating nBits
    def _get_param_names(self):
        """Get parameter names excluding deprecated parameters"""
        params = super()._get_param_names()
        # Remove deprecated parameters before they're accessed
        return [p for p in params if p not in ("nBits")]


class FpsGeneratorTransformer(BaseFpsTransformer):
    """Abstract base class for fingerprint transformers based on (unpicklable)fingerprint generators"""

    _regenerate_on_properties = ()

    def __getstate__(self):
        # Get the state of the parent class
        state = super().__getstate__()
        state.update(self.get_params())
        # Remove the potentiallyunpicklable property from the state
        state.pop("_fpgen", None)  # fpgen is not picklable
        return state

    def __setstate__(self, state):
        # Restore the state of the parent class
        super().__setstate__(state)
        # Re-create the unpicklable property
        generatort_keys = inspect.signature(
            self._generate_fp_generator
        ).parameters.keys()
        params = [
            setattr(self, k, state["_" + k])
            if "_" + k in state
            else setattr(self, k, state[k])
            for k in generatort_keys
        ]
        self._generate_fp_generator()

    # TODO: overload set_params in order to not make multiple calls to _generate_fp_generator

    def __setattr__(self, name: str, value):
        super().__setattr__(name, value)
        if (
            not hasattr(self, "_initializing")
            and name in self._regenerate_on_properties
        ):
            self._generate_fp_generator()

    @abstractmethod
    def _generate_fp_generator(self):
        raise NotImplementedError("_generate_fp_generator not implemented")

    @abstractmethod
    def _transform_mol(self, mol) -> np.array:
        """Generate numpy array descriptor from RDKit molecule

        MUST BE OVERWRITTEN
        """
        raise NotImplementedError("_transform_mol not implemented")

    # TODO, remove when finally deprecating nBits and dtype
    @property
    def dtype(self):
        warn(
            "dtype is no longer supported, due to move to generator based fingerprints",
            DeprecationWarning,
            stacklevel=2,
        )
        return None

    # TODO, remove when finally deprecating nBits and dtype
    @dtype.setter
    def dtype(self, dtype):
        if dtype is not None:
            warn(
                "dtype is no longer supported, due to move to generator based fingerprints",
                DeprecationWarning,
                stacklevel=3,
            )
        pass

    # TODO, remove when finally deprecating nBits and dtype
    def _get_param_names(self):
        """Get parameter names excluding deprecated parameters"""
        params = super()._get_param_names()
        # Remove deprecated parameters before they're accessed
        return [p for p in params if p not in ("dtype", "nBits")]


def parallel_helper(args):
    """Parallel_helper takes a tuple with classname, the objects parameters and the mols to process.
    Then instantiates the class with the parameters and processes the mol.
    Intention is to be able to do this in child processes as some classes can't be pickled"""
    classname, parameters, X_mols = args
    from scikit_mol import fingerprints

    transformer = getattr(fingerprints, classname)(**parameters)
    return transformer._transform(X_mols)
