from multiprocessing import Pool, get_context
import multiprocessing
import numpy as np
from typing import List, Optional, Any, Union

from rdkit.Chem.rdchem import Mol
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

from sklearn.base import BaseEstimator, TransformerMixin


class Desc2DTransformer(BaseEstimator, TransformerMixin):
    """Descriptor calculation transformer
    
    Parameters
    ----------
    desc_list : (List of descriptor names)
        A list of RDKit descriptors to include in the calculation
    parallel : boolean, int
        if True, multiprocessing will be used. If set to an int > 1, that specified number of processes
        will be used, otherwise it's autodetected.

    Returns
    -------
    np.array
        Descriptor values, shape (samples, len(descriptor list))


    """
    def __init__(
        self, desc_list: Optional[str] = None, 
        parallel: Union[bool, int] = False,
        start_method: str = "spawn"
        ):
        self.desc_list = desc_list
        self.parallel = parallel
        self.start_method = start_method

    def _get_desc_calculator(self) -> MolecularDescriptorCalculator:
        if self.desc_list:
            unknown_descriptors = [
                desc_name
                for desc_name in self.desc_list
                if desc_name not in self.available_descriptors
            ]
            assert (
                not unknown_descriptors
            ), f"Unknown descriptor names {unknown_descriptors} specified, please check available_descriptors property\nPlease check availble list {self.available_descriptors}"
        else:
            self.desc_list = self.available_descriptors
        return MolecularDescriptorCalculator(self.desc_list)

    @property
    def desc_list(self):
        """Descriptor names of currently selected descriptors"""
        return self._desc_list

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
        """Allowed methods are spawn, fork and forkserver on MacOS and Linux, only spawn is possible on Windows"""
        allowed_start_methods = ["spawn", "fork", "forkserver"]
        assert start_method in allowed_start_methods, f"start_method not in allowed methods {allowed_start_methods}"
        self._start_method = start_method

    def _transform_mol(self, mol: Mol) -> List[Any]:
        return list(self.calculators.CalcDescriptors(mol))

    def fit(self, x, y=None):
        """Included for scikit-learn compatibility, does nothing"""
        return self

    def _transform(self, x: List[Mol]) -> np.ndarray:
        arr = np.zeros((len(x), len(self.desc_list)))
        for i, mol in enumerate(x):
            arr[i, :] = self._transform_mol(mol)
        return arr

    def transform(self, x: List[Mol], y=None) -> np.ndarray:
        """Transform a list of molecules into an array of descriptor values
        Parameters
        ----------
        X : (List, np.array, pd.Series)
            A list of RDKit molecules
        y : NoneType, optional
            Target values for scikit-learn compatibility, not used, by default None

        Returns
        -------
        np.array
            Descriptors, shape (samples, length of .selected_descriptors )
        
        """
        if not self.parallel:
            return self._transform(x)
        elif self.parallel:
            n_processes = self.parallel if self.parallel > 1 else None #Pool(processes=None) autodetects
            n_chunks = n_processes if n_processes is not None else multiprocessing.cpu_count()
            with get_context(self.start_method).Pool(processes=n_processes) as pool:
                # pool = Pool(processes=self.n_processes)
                x_chunks = np.array_split(x, n_chunks * 3)  #TODO fix, n_processes may not be int, but None# Is x3 the optimal?
                arrays = pool.map(self._parallel_helper, [(self, x) for x in x_chunks]) #is the helper function a safer way of handling the picklind and child process communication
                # arrays = async_obj.get(20)
                #    pool.close()
                arr = np.concatenate(arrays)
            return arr

    @staticmethod
    def _parallel_helper(args):
        obj, x = args
        return obj._transform(x)
