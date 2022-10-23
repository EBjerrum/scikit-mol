import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

@pytest.fixture
def smiles_list():
    return [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in  ['O=C(O)c1ccccc1',
                'O=C([O-])c1ccccc1',
                'O=C([O-])c1ccccc1.[Na+]',
                'O=C(O[Na])c1ccccc1',
                'C[N+](C)C.O=C([O-])c1ccccc1']]

@pytest.fixture
def chiral_smiles_list():
    return [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in  [
                'N[C@@H](C)C(=O)O',
                'C1C[C@H]2CCCC[C@H]2CC1']]

@pytest.fixture
def invalid_smiles_list(smiles_list):
    smiles_list.append('Invalid')
    return smiles_list

@pytest.fixture
def mols_list(smiles_list):
    return [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

@pytest.fixture
def chiral_mols_list(chiral_smiles_list):
    return [Chem.MolFromSmiles(smiles) for smiles in chiral_smiles_list]


@pytest.fixture
def fingerprint(mols_list):
    return rdMolDescriptors.GetHashedMorganFingerprint(mols_list[0],2,nBits=1000)
