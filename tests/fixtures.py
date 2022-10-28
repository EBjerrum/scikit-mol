import os
import pytest
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

#TODO these should really go into the conftest.py, so that they are automatically imported in the tests

@pytest.fixture
def smiles_list():
    return [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in  ['O=C(O)c1ccccc1',
                'O=C([O-])c1ccccc1',
                'O=C([O-])c1ccccc1.[Na+]',
                'O=C(O[Na])c1ccccc1',
                'C[N+](C)C.O=C([O-])c1ccccc1']]

@pytest.fixture
def invalid_smiles_list(smiles_list):
    smiles_list.append('Invalid')
    return smiles_list

@pytest.fixture
def mols_list(smiles_list):
    return [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

@pytest.fixture
def fingerprint(mols_list):
    return rdMolDescriptors.GetHashedMorganFingerprint(mols_list[0],2,nBits=1000)

@pytest.fixture
def SLC6A4_subset():
    file_path = os.path.realpath(__file__)
    data = pd.read_csv(f"{os.path.split(file_path)[0]}/data/SLC6A4_active_excapedb_subset.csv")
    return data