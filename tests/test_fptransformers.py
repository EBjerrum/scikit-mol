import pytest
import numpy as np
import pandas as pd
from rdkit import Chem
from fixtures import mols_list, smiles_list, fingerprint
from sklearn import clone
from scikit_mol.transformers import MorganTransformer, MACCSTransformer, RDKitFPTransformer, AtomPairFingerprintTransformer, TopologicalTorsionFingerprintTransformer


@pytest.fixture
def morgan_transformer():
    return MorganTransformer()

@pytest.fixture
def rdkit_transformer():
    return RDKitFPTransformer()

@pytest.fixture
def atompair_transformer():
    return AtomPairFingerprintTransformer()

@pytest.fixture
def topologicaltorsion_transformer():
    return TopologicalTorsionFingerprintTransformer()

@pytest.fixture
def maccs_transformer():
    return MACCSTransformer()


def test_fpstransformer_fp2array(morgan_transformer, fingerprint):
    fp = morgan_transformer._fp2array(fingerprint)
    #See that fp is the correct type, shape and bit count
    assert(type(fp) == type(np.array([0])))
    assert(fp.shape == (1000,))
    assert(fp.sum() == 25)

def test_fpstransformer_transform_mol(morgan_transformer, mols_list):
    fp = morgan_transformer._transform_mol(mols_list[0])
    #See that fp is the correct type, shape and bit count
    assert(type(fp) == type(np.array([0])))
    assert(fp.shape == (2048,))
    assert(fp.sum() == 14)

def test_clonability(maccs_transformer, morgan_transformer, rdkit_transformer, atompair_transformer, topologicaltorsion_transformer):
    for t in [maccs_transformer, morgan_transformer, rdkit_transformer, atompair_transformer, topologicaltorsion_transformer]:
        params   = t.get_params()
        t2 = clone(t)
        params_2 = t2.get_params()
        #Parameters of cloned transformers should be the same
        assert all([ params[key] == params_2[key] for key in params.keys()])
        #Cloned transformers should not be the same object
        assert t2 != t

def test_set_params(morgan_transformer, rdkit_transformer, atompair_transformer, topologicaltorsion_transformer):
    for t in [morgan_transformer, atompair_transformer, topologicaltorsion_transformer]:
        params   = t.get_params()
        #change extracted dictionary
        params['nBits'] = 4242
        #change params in transformer
        t.set_params(nBits = 4242)
        # get parameters as dictionary and assert that it is the same
        params_2 = t.get_params()
        assert all([ params[key] == params_2[key] for key in params.keys()])

    for t in [rdkit_transformer]:
        params   = t.get_params()
        params['fpSize'] = 4242
        t.set_params(fpSize = 4242)
        params_2 = t.get_params()
        assert all([ params[key] == params_2[key] for key in params.keys()])

def test_transform(mols_list, morgan_transformer, rdkit_transformer, atompair_transformer, topologicaltorsion_transformer, maccs_transformer):
    #Test different types of input
    for mols in [mols_list, np.array(mols_list), pd.Series(mols_list)]:
        #Test the different transformers
        for t in [morgan_transformer, atompair_transformer, topologicaltorsion_transformer, maccs_transformer, rdkit_transformer]:
            params   = t.get_params()
            fps = t.transform(mols)
            #Assert that the same length of input and output
            assert len(fps) == len(mols_list)

            # assert that the size of the fingerprint is the expected size
            if type(t) == type(maccs_transformer):
                fpsize = t.nBits
            elif type(t) == type(rdkit_transformer):
                fpsize = params['fpSize']
            else:
                fpsize = params['nBits']
            
            assert len(fps[0]) == fpsize
