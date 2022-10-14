import numpy as np
from typing import List, Optional, Any

from rdkit.Chem.rdchem import Mol
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

from sklearn.base import BaseEstimator, TransformerMixin

class Desc2DTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self, use_all_descriptors: bool = True, custom_desc_list: Optional[str] = None
    ):
        self.use_all_descriptors = use_all_descriptors
        self.desc_list = custom_desc_list
        self.calculators = self._get_desc_calculator()

    def _get_desc_calculator(self) -> MolecularDescriptorCalculator:
        all_decriptors = [x[0] for x in Descriptors._descList]
        if self.use_all_descriptors:
            self.desc_list = all_decriptors
        else: 
            assert self.desc_list is not None, \
                "please provide your preferred descriptor names, otherwise let use_all_descriptors == True"
            self.desc_list = [desc for desc in self.desc_list if desc in all_decriptors]
        return MolecularDescriptorCalculator(self.desc_list)

    def get_desc_names(self) -> List[str]:
        # callable function to get list of descriptor names
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
