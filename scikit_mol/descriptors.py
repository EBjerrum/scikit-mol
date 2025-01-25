import functools
from typing import List, Optional, Union

import numpy as np
from rdkit.Chem import Descriptors
from rdkit.Chem.rdchem import Mol
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from sklearn.base import BaseEstimator, TransformerMixin

from scikit_mol.core import NoFitNeededMixin, check_transform_input
from scikit_mol.parallel import parallelized_with_batches


class MolecularDescriptorTransformer(TransformerMixin, NoFitNeededMixin, BaseEstimator):
    """Descriptor calculation transformer

    Parameters
    ----------
    desc_list : (List of descriptor names)
        A list of RDKit descriptors to include in the calculation
    parallel : boolean, int
        if True, multiprocessing will be used. If set to an int > 1, that specified number of processes
        will be used, otherwise it's autodetected.
    start_method : str
        The method to start child processes when parallel=True. can be 'fork', 'spawn' or 'forkserver'.
        If None, the OS and Pythons default will be used.
    safe_inference_mode : bool
        If True, enables safeguards for handling invalid data during inference.
        This should only be set to True when deploying models to production.

    Returns
    -------
    np.array
        Descriptor values, shape (samples, len(descriptor list))


    """

    def __init__(
        self,
        desc_list: Optional[str] = None,
        parallel: Optional[int] = None,
        start_method: Optional[str] = None,  # "fork",
        safe_inference_mode: bool = False,
        dtype: np.dtype = np.float32,
    ):
        self.desc_list = desc_list
        self.parallel = parallel
        self.start_method = start_method
        self.safe_inference_mode = safe_inference_mode
        self.dtype = dtype

    def _get_desc_calculator(self) -> MolecularDescriptorCalculator:
        if self.desc_list:
            unknown_descriptors = [
                desc_name
                for desc_name in self.desc_list
                if desc_name not in self.available_descriptors
            ]
            assert not unknown_descriptors, f"Unknown descriptor names {unknown_descriptors} specified, please check available_descriptors property\nPlease check availble list {self.available_descriptors}"
        else:
            self.desc_list = self.available_descriptors
        return MolecularDescriptorCalculator(self.desc_list)

    @property
    def desc_list(self):
        """Descriptor names of currently selected descriptors"""
        return self._desc_list

    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_descriptors)

    @desc_list.setter
    def desc_list(self, desc_list):
        self._desc_list = desc_list
        self.calculators = self._get_desc_calculator()

    @property
    def available_descriptors(self) -> List[str]:
        """List of names of all available descriptor"""
        return [descriptor[0] for descriptor in Descriptors._descList]

    @property
    def selected_descriptors(self) -> List[str]:
        """List of the names of the descriptors in the descriptor calculator"""
        return list(self.calculators.GetDescriptorNames())

    @property
    def start_method(self):
        return self._start_method

    @start_method.setter
    def start_method(self, start_method):
        """Allowed methods are spawn, fork and forkserver on MacOS and Linux, only spawn is possible on Windows.
        None will choose the default for the OS and version of Python."""
        allowed_start_methods = ["spawn", "fork", "forkserver", None]
        assert (
            start_method in allowed_start_methods
        ), f"start_method not in allowed methods {allowed_start_methods}"
        self._start_method = start_method

    def _transform_mol(self, mol: Mol) -> Union[np.ndarray, np.ma.MaskedArray]:
        if not mol:
            if self.safe_inference_mode:
                return np.ma.masked_all(len(self.desc_list))
            else:
                raise ValueError(f"Invalid molecule provided: {mol}")
        try:
            return np.array(list(self.calculators.CalcDescriptors(mol)))
        except Exception as e:
            if self.safe_inference_mode:
                return np.ma.masked_all(len(self.desc_list))
            else:
                raise e

    def fit(self, x, y=None):
        """Included for scikit-learn compatibility, does nothing"""
        return self

    @check_transform_input
    def _transform(self, x: List[Mol]) -> Union[np.ndarray, np.ma.MaskedArray]:
        if self.safe_inference_mode:
            arrays = [self._transform_mol(mol) for mol in x]
            return np.ma.array(arrays, dtype=self.dtype)
        else:
            arr = np.zeros((len(x), len(self.desc_list)), dtype=self.dtype)
            for i, mol in enumerate(x):
                arr[i, :] = self._transform_mol(mol)
            return arr

    def transform(self, x: List[Mol], y=None) -> Union[np.ndarray, np.ma.MaskedArray]:
        """Transform a list of molecules into an array of descriptor values
        Parameters
        ----------
        X : (List, np.array, pd.Series)
            A list of RDKit molecules
        y : NoneType, optional
            Target values for scikit-learn compatibility, not used, by default None

        Returns
        -------
        Union[np.ndarray, np.ma.MaskedArray]
            Descriptors, shape (samples, length of .selected_descriptors)

        """
        fn = functools.partial(parallel_helper, self.get_params())
        arrays = parallelized_with_batches(fn, x, self.parallel)
        if self.safe_inference_mode:
            arrays = np.ma.concatenate(arrays)
        else:
            arrays = np.concatenate(arrays)
        return arrays


# May be safer to instantiate the transformer object in the child process, and only transfer the parameters
# There were issues with freezing when using RDKit 2022.3
def parallel_helper(params, mols):
    """Will get a tuple with Desc2DTransformer parameters and mols to transform.
    Will then instantiate the transformer and transform the molecules"""
    from scikit_mol.descriptors import MolecularDescriptorTransformer

    transformer = MolecularDescriptorTransformer(**params)
    y = transformer._transform(mols)
    return y
