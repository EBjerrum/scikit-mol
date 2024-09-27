import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from fixtures import smiles_list, invalid_smiles_list
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.fingerprints import MorganFingerprintTransformer
from scikit_mol.wrapper import WrappedTransformer
from scikit_mol._invalid import NumpyArrayWithInvalidInstances
from test_invalid_helpers.invalid_transformer import TestInvalidTransformer


@pytest.fixture
def smilestofp_pipeline():
    pipeline = Pipeline(
        [
            ("smiles_to_mol", SmilesToMolTransformer()),
            ("remove_sulfur", TestInvalidTransformer()),
            ("mol_2_fp", MorganFingerprintTransformer()),
            ("PCA", WrappedTransformer(PCA(2), replace_invalid=True)),
        ]
    )
    return pipeline


def test_descriptor_transformer(smiles_list, invalid_smiles_list, smilestofp_pipeline):
    smilestofp_pipeline.set_params()
    mol_pca = smilestofp_pipeline.fit_transform(smiles_list)
    error_mol_pca = smilestofp_pipeline.fit_transform(invalid_smiles_list)

    if mol_pca.shape != (len(smiles_list), 2):
        raise ValueError("The PCA does not return the proper dimensions.")
    if isinstance(error_mol_pca, NumpyArrayWithInvalidInstances):
        raise TypeError("The Errors were not properly remove from the output array.")

    expected_nans = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    if not np.all(np.equal(expected_nans, np.where(np.isnan(error_mol_pca)))):
        raise ValueError("Errors were replaced on the wrong positions.")

    non_nan_rows = ~np.any(np.isnan(error_mol_pca), axis=1)
    if not np.all(np.isclose(mol_pca, error_mol_pca[non_nan_rows, :])):
        raise ValueError("Removing errors introduces changes in the PCA output.")
