import pytest
from rdkit import Chem

@pytest.fixture
def smiles_list():
    return [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in  ['O=C(O)c1ccccc1',
                'O=C([O-])c1ccccc1',
                'O=C([O-])c1ccccc1.[Na+]',
                'O=C(O[Na])c1ccccc1',
                'C[N+](C)C.O=C([O-])c1ccccc1']]

@pytest.fixture
def invalid_smiles_list(smiles_list):
    smiles_list.append('Invalid')
    return smiles_list

@pytest.fixture
def mols_list(smiles_list):
    return [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
