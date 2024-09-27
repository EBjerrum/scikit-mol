import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from fixtures import smiles_list, invalid_smiles_list
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.fingerprints import (
    MorganFingerprintTransformer,
    MACCSKeysFingerprintTransformer,
)
from scikit_mol.wrapper import NanGuardWrapper  # WrappedTransformer

# from scikit_mol._invalid import NumpyArrayWithInvalidInstances
# from test_invalid_helpers.invalid_transformer import TestInvalidTransformer


@pytest.fixture
def smilestofp_pipeline():
    pipeline = Pipeline(
        [
            ("smiles_to_mol", SmilesToMolTransformer(handle_errors=True)),
            ("mol_2_fp", MACCSKeysFingerprintTransformer(handle_errors=True)),
            ("PCA", NanGuardWrapper(PCA(2), handle_errors=True)),
        ]
    )
    return pipeline


def test_descriptor_transformer(smiles_list, invalid_smiles_list, smilestofp_pipeline):
    # smilestofp_pipeline.set_params()
    mol_pca = smilestofp_pipeline.fit_transform(smiles_list)
    error_mol_pca = smilestofp_pipeline.fit_transform(invalid_smiles_list)

    print(mol_pca.shape)
    assert mol_pca.shape == (
        len(smiles_list),
        2,
    ), "The PCA does not return the proper dimensions."

    expected_nans = np.array([[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]]).T
    if not np.all(np.equal(expected_nans, np.isnan(error_mol_pca))):
        raise ValueError("Errors were replaced on the wrong positions.")

    non_nan_rows = ~np.any(np.isnan(error_mol_pca), axis=1)
    assert np.all(
        np.isclose(mol_pca, error_mol_pca[non_nan_rows, :])
    ), "Removing errors introduces changes in the PCA output."

    # TODO, test with and without error handling on
    # TODO, test with other transformers
