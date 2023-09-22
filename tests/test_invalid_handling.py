import pytest
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.fingerprints import MorganFingerprintTransformer
from sklearn.pipeline import Pipeline
from fixtures import smiles_list, invalid_smiles_list
from scikit_mol._invalid import ArrayWithInvalidInstances

@pytest.fixture
def smilestofp_pipeline():
    pipeline = Pipeline(
        [
            ("smiles_to_mol", SmilesToMolTransformer()),
            ("mol_2_fp", MorganFingerprintTransformer()),
        ]

    )
    return pipeline


def test_descriptor_transformer(invalid_smiles_list, smilestofp_pipeline):
    smilestofp_pipeline.set_params()
    mol_list: ArrayWithInvalidInstances = smilestofp_pipeline.transform(invalid_smiles_list)
    print(mol_list.is_valid_array)
    print(mol_list.matrix)
    print(mol_list.invalid_list)
