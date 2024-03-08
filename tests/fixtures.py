import os
import pytest
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from packaging.version import Version
import sklearn

from scikit_mol.core import SKLEARN_VERSION_PANDAS_OUT

#TODO these should really go into the conftest.py, so that they are automatically imported in the tests

_SMILES_LIST = [
    'O=C(O)c1ccccc1',
    'O=C([O-])c1ccccc1',
    'O=C([O-])c1ccccc1.[Na+]',
    'O=C(O[Na])c1ccccc1',
    'C[N+](C)C.O=C([O-])c1ccccc1',
]
_CANONICAL_SMILES_LIST = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in  _SMILES_LIST]

@pytest.fixture
def smiles_list():
    return _CANONICAL_SMILES_LIST.copy()

_CONTAINER_CREATORS = [
    lambda x: x,
    lambda x: np.array(x),
    lambda x: np.array(x).reshape(-1, 1),
]
_names_to_test = [
    "molecule",
    "mol",
    "smiles",
    "ROMol",
    "hello",
    None,
]
for name in _names_to_test:
    _CONTAINER_CREATORS.append(lambda x: pd.Series(x, name=name))
    _CONTAINER_CREATORS.append(lambda x: pd.DataFrame({name: x}) if name else pd.DataFrame(x))

@pytest.fixture(params=[container(_CANONICAL_SMILES_LIST) for container in _CONTAINER_CREATORS]
)
def smiles_container(request, ):
    return request.param.copy()

@pytest.fixture
def chiral_smiles_list(): #Need to be a certain size, so the fingerprints reacts to different max_lenǵths and radii
    return [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in  [
                'N[C@@H](C)C(=O)OCCCCCCCCCCCC',
                'C1C[C@H]2CCCC[C@H]2CC1CCCCCCCCC',
                'N[C@@H](C)C(=O)Oc1ccccc1CCCCCCCCCCCCCCCCCCN[H]']]

@pytest.fixture
def invalid_smiles_list(smiles_list):
    smiles_list.append('Invalid')
    return smiles_list

_MOLS_LIST = [Chem.MolFromSmiles(smiles) for smiles in _SMILES_LIST]

@pytest.fixture
def mols_list():
    return _MOLS_LIST.copy()

@pytest.fixture(params=[container(_MOLS_LIST) for container in _CONTAINER_CREATORS])
def mols_container(request):
    return request.param.copy()

@pytest.fixture
def chiral_mols_list(chiral_smiles_list):
    return [Chem.MolFromSmiles(smiles) for smiles in chiral_smiles_list]


@pytest.fixture
def fingerprint(mols_list):
    return rdMolDescriptors.GetHashedMorganFingerprint(mols_list[0],2,nBits=1000)

@pytest.fixture
def SLC6A4_subset():
    file_path = os.path.realpath(__file__)
    data = pd.read_csv(f"{os.path.split(file_path)[0]}/data/SLC6A4_active_excapedb_subset.csv")
    return data

skip_pandas_output_test = pytest.mark.skipif(Version(sklearn.__version__) < SKLEARN_VERSION_PANDAS_OUT, reason=f"requires scikit-learn {SKLEARN_VERSION_PANDAS_OUT} or higher")