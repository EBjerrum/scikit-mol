import pytest
import numpy as np
import pandas as pd
from packaging.version import Version
from sklearn import clone
from rdkit import Chem
import sklearn
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.core import SKLEARN_VERSION_PANDAS_OUT, DEFAULT_MOL_COLUMN_NAME
from fixtures import (
    smiles_list,
    invalid_smiles_list,
    smiles_container,
    skip_pandas_output_test,
)


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
    assert all([a == b for a, b in zip(expected_smiles, result_smiles)])


def test_smilestomol_transform(smilestomol_transformer, smiles_container):
    result = smilestomol_transformer.transform(smiles_container)
    assert len(result) == len(smiles_container)
    assert all(isinstance(mol, Chem.Mol) for mol in result.flatten())


def test_smilestomol_fit(smilestomol_transformer, smiles_container):
    result = smilestomol_transformer.fit(smiles_container)
    assert result == smilestomol_transformer


def test_smilestomol_clone(smilestomol_transformer):
    t2 = clone(smilestomol_transformer)
    params = smilestomol_transformer.get_params()
    params_2 = t2.get_params()
    assert all([params[key] == params_2[key] for key in params.keys()])


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
    assert all(
        [
            a == b
            for a, b in zip(
                expected_smiles, [Chem.MolToSmiles(mol) for mol in mol_list.flatten()]
            )
        ]
    )


def test_smilestomol_inverse_transform(smilestomol_transformer, smiles_container):
    mols = smilestomol_transformer.transform(smiles_container)
    result = smilestomol_transformer.inverse_transform(mols)
    assert len(result) == len(smiles_container)
    assert all(isinstance(smiles, str) for smiles in result.flatten())


@skip_pandas_output_test
def test_pandas_output(smiles_container, smilestomol_transformer, pandas_output):
    mols = smilestomol_transformer.transform(smiles_container)
    assert isinstance(mols, pd.DataFrame)
    assert mols.shape[0] == len(smiles_container)
    assert mols.columns.tolist() == [DEFAULT_MOL_COLUMN_NAME]


def test_smilestomol_get_feature_names_out(smilestomol_transformer):
    feature_names = smilestomol_transformer.get_feature_names_out()
    assert feature_names == [DEFAULT_MOL_COLUMN_NAME]
