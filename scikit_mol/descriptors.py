import numpy as np
from typing import List, Optional, Any

from rdkit.Chem.rdchem import Mol
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

from sklearn.base import BaseEstimator, TransformerMixin

class Desc2DTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self, desc_list: Optional[str] = None
    ):
        self.desc_list = desc_list

    def _get_desc_calculator(self) -> MolecularDescriptorCalculator:
        if self.desc_list:
            unknown_descriptors = [desc_name for desc_name in self.desc_list if desc_name not in self.available_descriptors]
            assert (not unknown_descriptors), f"Unknown descriptor names {unknown_descriptors} specified, please check available_descriptors property\nPlease check availble list {self.available_descriptors}" 
        else:
            self.desc_list = self.available_descriptors 
        return MolecularDescriptorCalculator(self.desc_list)

    @property
    def desc_list(self):
        return self._desc_list

    @desc_list.setter
    def desc_list(self, desc_list):
        self._desc_list = desc_list
        self.calculators = self._get_desc_calculator()

    @property
    def available_descriptors(self) -> List[str]:
        """Property to get list of all available descriptor names"""
        return [descriptor[0] for descriptor in Descriptors._descList]

    @property
    def selected_descriptors(self) -> List[str]:
        """Property to get list of the selected descriptor names"""
        return list(self.calculators.GetDescriptorNames())

    def _transform_mol(self, mol: Mol) -> List[Any]:
        return list(self.calculators.CalcDescriptors(mol))

    def fit(self, x, y=None):
        return self

    def transform(self, x: List[Mol], y=None) -> np.ndarray:
        arr = np.zeros((len(x), len(self.desc_list)))  
        for i, mol in enumerate(x):
            arr[i, :] = self._transform_mol(mol)
        return arr
