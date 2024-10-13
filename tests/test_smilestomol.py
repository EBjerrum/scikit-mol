import pytest
import numpy as np
import pandas as pd
from packaging.version import Version
from sklearn import clone
from rdkit import Chem
import sklearn
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.core import (
    SKLEARN_VERSION_PANDAS_OUT,
    DEFAULT_MOL_COLUMN_NAME,
    InvalidMol,
)
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


def test_smilestomol_inverse_transform_with_invalid(
    invalid_smiles_list, smilestomol_transformer
):
    smilestomol_transformer.set_params(safe_inference_mode=True)

    # Forward transform
    mols = smilestomol_transformer.transform(invalid_smiles_list)

    # Inverse transform
    result = smilestomol_transformer.inverse_transform(mols)

    assert len(result) == len(invalid_smiles_list)

    # Check that all but the last element are the same as the original SMILES
    for original, res in zip(invalid_smiles_list[:-1], result[:-1].flatten()):
        assert isinstance(res, str)
        assert original == res

    # Check that the last element is an InvalidMol instance
    assert isinstance(result[-1].item(), InvalidMol)
    assert "Invalid SMILES" in result[-1].item().error
    assert invalid_smiles_list[-1] in result[-1].item().error


def test_smilestomol_get_feature_names_out(smilestomol_transformer):
    feature_names = smilestomol_transformer.get_feature_names_out()
    assert feature_names == [DEFAULT_MOL_COLUMN_NAME]


def test_smilestomol_safe_inference(invalid_smiles_list, smilestomol_transformer):
    smilestomol_transformer.set_params(safe_inference_mode=True)
    result = smilestomol_transformer.transform(invalid_smiles_list)

    assert len(result) == len(invalid_smiles_list)
    assert isinstance(result, np.ndarray)

    # Check that all but the last element are valid RDKit Mol objects
    for mol in result[:-1].flatten():
        assert isinstance(mol, Chem.Mol)
        assert mol is not None

    # Check that the last element is an InvalidMol instance
    last_mol = result[-1].item()
    assert isinstance(last_mol, InvalidMol)

    # Check if the error message is correctly set for the invalid SMILES
    assert "Invalid SMILES" in last_mol.error
    assert invalid_smiles_list[-1] in last_mol.error


@pytest.mark.skipif(
    not skip_pandas_output_test,
    reason="Pandas output not supported in this sklearn version",
)
def test_smilestomol_safe_inference_pandas_output(
    invalid_smiles_list, smilestomol_transformer, pandas_output
):
    smilestomol_transformer.set_params(safe_inference_mode=True)
    result = smilestomol_transformer.transform(invalid_smiles_list)

    assert len(result) == len(invalid_smiles_list)
    assert isinstance(result, pd.DataFrame)
    assert result.columns == [DEFAULT_MOL_COLUMN_NAME]

    # Check that all but the last element are valid RDKit Mol objects
    for mol in result[DEFAULT_MOL_COLUMN_NAME][:-1]:
        assert isinstance(mol, Chem.Mol)
        assert mol is not None

    # Check that the last element is an InvalidMol instance
    last_mol = result[DEFAULT_MOL_COLUMN_NAME].iloc[-1]
    assert isinstance(last_mol, InvalidMol)

    # Check if the error message is correctly set for the invalid SMILES
    assert "Invalid SMILES" in last_mol.error
    assert invalid_smiles_list[-1] in last_mol.error


@skip_pandas_output_test
def test_pandas_output(smiles_container, smilestomol_transformer, pandas_output):
    mols = smilestomol_transformer.transform(smiles_container)
    assert isinstance(mols, pd.DataFrame)
    assert mols.shape[0] == len(smiles_container)
    assert mols.columns.tolist() == [DEFAULT_MOL_COLUMN_NAME]
