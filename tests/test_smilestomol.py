import pytest
import numpy as np
import pandas as pd
from sklearn import clone
from rdkit import Chem
from scikit_mol.transformers import SmilesToMol
from fixtures import smiles_list, invalid_smiles_list


@pytest.fixture
def smilestomol_transformer():
    return SmilesToMol()

def test_smilestomol(smiles_list, smilestomol_transformer):
    mol_list = smilestomol_transformer.transform(smiles_list)
    assert all([ a == b for a, b in zip(smiles_list, [Chem.MolToSmiles(mol) for mol in mol_list])])

def test_smilestomol_numpy(smiles_list, smilestomol_transformer):
    mol_list = smilestomol_transformer.transform(np.array(smiles_list))

    assert all([ a == b for a, b in zip(smiles_list, [Chem.MolToSmiles(mol) for mol in mol_list])])

def test_smilestomol_pandas(smiles_list, smilestomol_transformer):
    mol_list = smilestomol_transformer.transform(pd.Series(smiles_list))

    assert all([ a == b for a, b in zip(smiles_list, [Chem.MolToSmiles(mol) for mol in mol_list])])

def test_smilestomol_clone(smilestomol_transformer):
    t2 = clone(smilestomol_transformer)
    params   = smilestomol_transformer.get_params()
    params_2 = t2.get_params()
    assert all([ params[key] == params_2[key] for key in params.keys()])

def test_smilestomol_unsanitzable(invalid_smiles_list, smilestomol_transformer):
    with pytest.raises(ValueError):
        smilestomol_transformer.transform(invalid_smiles_list)

    