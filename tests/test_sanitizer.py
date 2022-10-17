import pytest
import numpy as np
import pandas as pd
from rdkit import Chem
from fixtures import smiles_list, invalid_smiles_list
from scikit_mol.sanitizer import CheckSmilesSanitazion

@pytest.fixture
def sanitizer():
    return CheckSmilesSanitazion()

@pytest.fixture
def return_mol_sanitizer():
    return CheckSmilesSanitazion(return_mol=True)

def test_checksmilessanitation(smiles_list, invalid_smiles_list, sanitizer):
    smiles_list_sanitized, errors = sanitizer.sanitize(invalid_smiles_list)
    assert len(invalid_smiles_list) > len(smiles_list_sanitized)
    assert all([ a == b for a, b in zip(smiles_list, smiles_list_sanitized)])
    assert errors[0] == sanitizer.errors.SMILES[0]

def test_checksmilessanitation_np(smiles_list, invalid_smiles_list, sanitizer):
    smiles_list_sanitized, errors = sanitizer.sanitize(np.array(invalid_smiles_list))
    assert len(invalid_smiles_list) > len(smiles_list_sanitized)
    assert all([ a == b for a, b in zip(smiles_list, smiles_list_sanitized)])
    assert errors[0] == sanitizer.errors.SMILES[0]

def test_checksmilessanitation_numpy(smiles_list, invalid_smiles_list, sanitizer):
    smiles_list_sanitized, errors = sanitizer.sanitize(pd.Series(invalid_smiles_list))
    assert len(invalid_smiles_list) > len(smiles_list_sanitized)
    assert all([ a == b for a, b in zip(smiles_list, smiles_list_sanitized)])
    assert errors[0] == sanitizer.errors.SMILES[0]

def test_checksmilessanitation_return_mol(smiles_list, invalid_smiles_list, return_mol_sanitizer):
    smiles_list_sanitized, errors = return_mol_sanitizer.sanitize(invalid_smiles_list)
    assert len(invalid_smiles_list) > len(smiles_list_sanitized)
    assert all([ a == b for a, b in zip(smiles_list, [Chem.MolToSmiles(smiles) for smiles in smiles_list_sanitized])])
    assert errors[0] == return_mol_sanitizer.errors.SMILES[0]