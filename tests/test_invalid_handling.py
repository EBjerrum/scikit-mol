import pytest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from fixtures import smiles_list, invalid_smiles_list
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.fingerprints import MorganFingerprintTransformer
from scikit_mol.wrapper import WrappedTransformer
from scikit_mol._invalid import NumpyArrayWithInvalidInstances

@pytest.fixture
def smilestofp_pipeline():
    pipeline = Pipeline(
        [
            ("smiles_to_mol", SmilesToMolTransformer()),
            ("mol_2_fp", MorganFingerprintTransformer()),
            ("PCA", WrappedTransformer(PCA(2), replace_invalid=True))
        ]

    )
    return pipeline


def test_descriptor_transformer(invalid_smiles_list, smilestofp_pipeline):
    smilestofp_pipeline.set_params()
    mol_list: NumpyArrayWithInvalidInstances = smilestofp_pipeline.fit_transform(invalid_smiles_list)
    print(mol_list)
