import pytest
import numpy as np
import pandas as pd
from rdkit import Chem
from scikit_mol.transformers import SmilesToMol

smiles_list = ['O=C(O)c1ccccc1',
                'O=C([O-])c1ccccc1',
                'O=C([O-])c1ccccc1.[Na+]',
                'O=C(O[Na])c1ccccc1',
                'C[N+](C)C.O=C([O-])c1ccccc1']

def test_smilestomol():
    transformer = SmilesToMol()

    mol_list = transformer.transform(smiles_list)

    assert all([ a == b for a, b in zip(smiles_list, [Chem.MolToSmiles(mol) for mol in mol_list])])

def test_smilestomol_numpy():
    transformer = SmilesToMol()

    mol_list = transformer.transform(np.array(smiles_list))

    assert all([ a == b for a, b in zip(smiles_list, [Chem.MolToSmiles(mol) for mol in mol_list])])

def test_smilestomol_pandas():
    transformer = SmilesToMol()

    mol_list = transformer.transform(pd.Series(smiles_list))

    assert all([ a == b for a, b in zip(smiles_list, [Chem.MolToSmiles(mol) for mol in mol_list])])


def test_smilestomol_unsanitzable():
    transformer = SmilesToMol()

    smiles_list = ['Invalid']
    
    with pytest.raises(ValueError):
        transformer.transform(smiles_list)

    