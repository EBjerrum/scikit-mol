import sys
import pytest
import numpy as np
import pandas as pd
from sklearn import clone
from rdkit import Chem
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.core import MIN_PYTHON_FOR_PANDAS_OUT
from fixtures import smiles_list, invalid_smiles_list, smiles_container


@pytest.fixture
def smilestomol_transformer():
    return SmilesToMolTransformer()

def test_smilestomol(smiles_container, smilestomol_transformer):
        result_mols = smilestomol_transformer.transform(smiles_container)
        result_smiles = [Chem.MolToSmiles(mol) for mol in result_mols.flatten()]
        if isinstance(smiles_container, pd.DataFrame):
            expected_smiles = smiles_container.iloc[:, 0].tolist()
        else:
            expected_smiles = smiles_container
        assert all([ a == b for a, b in zip(expected_smiles, result_smiles)])

def test_smilestomol_clone(smilestomol_transformer):
    t2 = clone(smilestomol_transformer)
    params   = smilestomol_transformer.get_params()
    params_2 = t2.get_params()
    assert all([ params[key] == params_2[key] for key in params.keys()])

def test_smilestomol_unsanitzable(invalid_smiles_list, smilestomol_transformer):
    with pytest.raises(ValueError):
        smilestomol_transformer.transform(invalid_smiles_list)

def test_descriptor_transformer_parallel(smiles_container, smilestomol_transformer):
    smilestomol_transformer.set_params(parallel=True)
    mol_list = smilestomol_transformer.transform(smiles_container)
    if isinstance(smiles_container, pd.DataFrame):
        expected_smiles = smiles_container.iloc[:, 0].tolist()
    else:
        expected_smiles = smiles_container
    assert all([ a == b for a, b in zip(expected_smiles, [Chem.MolToSmiles(mol) for mol in mol_list.flatten()])])

# TODO: see if it is possible to define this skipif condition
# as its own decorator / fixture, to avoid duplication, even though it is simple.
@pytest.mark.skipif(sys.version_info < MIN_PYTHON_FOR_PANDAS_OUT.as_tuple(), reason="requires Python 3.8 or higher")
def test_pandas_output(smiles_container, smilestomol_transformer, pandas_output):
        mols = smilestomol_transformer.transform(smiles_container)
        assert isinstance(mols, pd.DataFrame)
        assert mols.shape[0] == len(smiles_container)
        assert mols.columns.tolist() == ["ROMol"]