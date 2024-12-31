import pytest
import numpy as np
import pandas as pd
from rdkit import Chem
from fixtures import smiles_list, smiles_list_with_invalid
from scikit_mol.utilities import CheckSmilesSanitazion


@pytest.fixture
def sanitizer():
    return CheckSmilesSanitazion()


@pytest.fixture
def return_mol_sanitizer():
    return CheckSmilesSanitazion(return_mol=True)


def test_checksmilessanitation(smiles_list, smiles_list_with_invalid, sanitizer):
    smiles_list_sanitized, errors = sanitizer.sanitize(smiles_list_with_invalid)
    assert len(smiles_list_with_invalid) > len(smiles_list_sanitized)
    assert all([a == b for a, b in zip(smiles_list, smiles_list_sanitized)])
    assert errors[0] == sanitizer.errors.SMILES[0]


def test_checksmilessanitation_x_and_y(smiles_list, smiles_list_with_invalid, sanitizer):
    smiles_list_sanitized, y_sanitized, errors, y_errors = sanitizer.sanitize(
        smiles_list_with_invalid, list(range(len(smiles_list_with_invalid)))
    )
    assert len(smiles_list_with_invalid) > len(smiles_list_sanitized)
    assert all([a == b for a, b in zip(smiles_list, smiles_list_sanitized)])
    assert errors[0] == sanitizer.errors.SMILES[0]
    # Test that y is correctly split into y_error and the rest
    assert all([a == b for a, b in zip(y_sanitized, list(range(len(smiles_list) - 1)))])
    assert y_errors[0] == len(smiles_list_with_invalid) - 1  # Last smiles is invalid


def test_checksmilessanitation_np(smiles_list, smiles_list_with_invalid, sanitizer):
    smiles_list_sanitized, errors = sanitizer.sanitize(np.array(smiles_list_with_invalid))
    assert len(smiles_list_with_invalid) > len(smiles_list_sanitized)
    assert all([a == b for a, b in zip(smiles_list, smiles_list_sanitized)])
    assert errors[0] == sanitizer.errors.SMILES[0]


def test_checksmilessanitation_numpy(smiles_list, smiles_list_with_invalid, sanitizer):
    smiles_list_sanitized, errors = sanitizer.sanitize(pd.Series(smiles_list_with_invalid))
    assert len(smiles_list_with_invalid) > len(smiles_list_sanitized)
    assert all([a == b for a, b in zip(smiles_list, smiles_list_sanitized)])
    assert errors[0] == sanitizer.errors.SMILES[0]


def test_checksmilessanitation_return_mol(
    smiles_list, smiles_list_with_invalid, return_mol_sanitizer
):
    smiles_list_sanitized, errors = return_mol_sanitizer.sanitize(smiles_list_with_invalid)
    assert len(smiles_list_with_invalid) > len(smiles_list_sanitized)
    assert all(
        [
            a == b
            for a, b in zip(
                smiles_list,
                [Chem.MolToSmiles(smiles) for smiles in smiles_list_sanitized],
            )
        ]
    )
    assert errors[0] == return_mol_sanitizer.errors.SMILES[0]
