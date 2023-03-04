import time
import pytest 
import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors
from scikit_mol.descriptors import Desc2DTransformer
from fixtures import mols_list, smiles_list
from sklearn import clone
import joblib



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
        
def test_descriptor_transformer_wrong_descriptors():
    with pytest.raises(AssertionError):
        Desc2DTransformer(desc_list=['Color', 'Icecream content', 'ChokolateDarkness', 'Content42', 'MolWt'])



def test_descriptor_transformer_parallel(mols_list, default_descriptor_transformer):
    default_descriptor_transformer.set_params(parallel=True)
    features = default_descriptor_transformer.transform(mols_list)
    assert(len(features) == len(mols_list))
    assert(len(features[0]) == len(Descriptors._descList))
    #Now with Rdkit 2022.3 creating a second transformer and running it, froze the process
    transformer2 = Desc2DTransformer(**default_descriptor_transformer.get_params())
    features2 = transformer2.transform(mols_list)
    assert(len(features2) == len(mols_list))
    assert(len(features2[0]) == len(Descriptors._descList))

# This test may fail on windows and mac (due to spawn rather than fork?)
# def test_descriptor_transformer_parallel_speedup(mols_list, default_descriptor_transformer):
#     n_phys_cpus = joblib.cpu_count(only_physical_cores=True)
#     mols_list = mols_list*50
#     if n_phys_cpus > 1:
#         t0 = time.time()
#         features = default_descriptor_transformer.transform(mols_list)
#         t_single = time.time()-t0
        
#         default_descriptor_transformer.set_params(parallel=True)
#         t0 = time.time()
#         features = default_descriptor_transformer.transform(mols_list)
#         t_par = time.time()-t0

#         assert(t_par < t_single/(n_phys_cpus/1.5)) # div by 1.5 as we don't assume full speedup

    
        


