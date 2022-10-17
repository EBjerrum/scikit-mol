import pytest 
import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors
from scikit_mol.descriptors import Desc2DTransformer
from fixtures import mols_list, smiles_list
from sklearn import clone



@pytest.fixture
def default_descriptor_transformer():
    return Desc2DTransformer()

@pytest.fixture
def selected_descriptor_transformer():
    return Desc2DTransformer(desc_list=['HeavyAtomCount', 'FractionCSP3', 'RingCount', 'MolLogP', 'MolWt'])

def test_descriptor_transformer_clonability( default_descriptor_transformer):
    for t in [ default_descriptor_transformer]:
        params   = t.get_params()
        t2 = clone(t)
        params_2 = t2.get_params()
        #Parameters of cloned transformers should be the same
        assert all([ params[key] == params_2[key] for key in params.keys()])
        #Cloned transformers should not be the same object
        assert t2 != t

def test_descriptor_transformer_set_params(default_descriptor_transformer):
    for t in [default_descriptor_transformer]:
        params   = t.get_params()
        #change extracted dictionary
        params['desc_list'] = ['HeavyAtomCount', 'FractionCSP3']
        #change params in transformer
        t.set_params(desc_list = ['HeavyAtomCount', 'FractionCSP3'])
        # get parameters as dictionary and assert that it is the same
        params_2 = t.get_params()
        assert all([ params[key] == params_2[key] for key in params.keys()])
        assert len(default_descriptor_transformer.selected_descriptors) == 2

def test_descriptor_transformer_available_descriptors(default_descriptor_transformer, selected_descriptor_transformer):
    #Default have as many as in RDkit and all are selected
    assert (len(default_descriptor_transformer.available_descriptors) ==  len(Descriptors._descList))
    assert (len(default_descriptor_transformer.selected_descriptors) ==  len(Descriptors._descList))
    #Default have as many as in RDkit but only 5 are selected
    assert (len(selected_descriptor_transformer.available_descriptors) ==  len(Descriptors._descList))
    assert (len(selected_descriptor_transformer.selected_descriptors) ==  5)
    

def test_descriptor_transformer_transform(mols_list, default_descriptor_transformer):
    for mols in  [mols_list, np.array(mols_list), pd.Series(mols_list)]:
        features = default_descriptor_transformer.transform(mols)
        assert(len(features) == len(mols))
        assert(len(features[0]) == len(Descriptors._descList))






