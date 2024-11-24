import pickle
import tempfile
import pytest
import numpy as np
from fixtures import (
    mols_list,
    smiles_list,
    mols_container,
    smiles_container,
    fingerprint,
    chiral_smiles_list,
    chiral_mols_list,
)
from sklearn import clone

from scikit_mol.fingerprints import (
    AtomPairFingerprintTransformer,
    MorganFingerprintTransformer,
    RDKitFingerprintTransformer,
    TopologicalTorsionFingerprintTransformer,
)

test_transformers = [
    AtomPairFingerprintTransformer,
    MorganFingerprintTransformer,
    RDKitFingerprintTransformer,
    TopologicalTorsionFingerprintTransformer,
]


@pytest.mark.parametrize("transformer_class", test_transformers)
def test_fpstransformer_transform_mol(transformer_class, mols_list):
    transformer = transformer_class()

    fp = transformer._transform_mol(mols_list[0])
    # See that fp is the correct type, shape and bit count
    assert type(fp) == type(np.array([0]))
    assert fp.shape == (2048,)

    if isinstance(transformer, RDKitFingerprintTransformer):
        assert fp.sum() == 104
    elif isinstance(transformer, AtomPairFingerprintTransformer):
        assert fp.sum() == 32
    elif isinstance(transformer, TopologicalTorsionFingerprintTransformer):
        assert fp.sum() == 12
    elif isinstance(transformer, MorganFingerprintTransformer):
        assert fp.sum() == 14
    else:
        raise NotImplementedError(f"missing Assert for {transformer_class}")


@pytest.mark.parametrize("transformer_class", test_transformers)
def test_clonability(transformer_class):
    transformer = transformer_class()

    params = transformer.get_params()
    t2 = clone(transformer)
    params_2 = t2.get_params()
    # Parameters of cloned transformers should be the same
    assert all([params[key] == params_2[key] for key in params.keys()])
    # Cloned transformers should not be the same object
    assert t2 != transformer


@pytest.mark.parametrize("transformer_class", test_transformers)
def test_set_params(transformer_class):
    transformer = transformer_class()
    params = transformer.get_params()
    # change extracted dictionary
    params["fpSize"] = 4242
    # change params in transformer
    transformer.set_params(fpSize=4242)
    # get parameters as dictionary and assert that it is the same
    params_2 = transformer.get_params()
    assert all([params[key] == params_2[key] for key in params.keys()])


@pytest.mark.parametrize("transformer_class", test_transformers)
def test_transform(mols_container, transformer_class):
    transformer = transformer_class()
    # Test the different transformers
    params = transformer.get_params()
    fps = transformer.transform(mols_container)
    # Assert that the same length of input and output
    assert len(fps) == len(mols_container)

    fpsize = params["fpSize"]

    assert len(fps[0]) == fpsize


@pytest.mark.parametrize("transformer_class", test_transformers)
def test_transform_parallel(mols_container, transformer_class):
    transformer = transformer_class()
    # Test the different transformers
    transformer.set_params(parallel=True)
    params = transformer.get_params()
    fps = transformer.transform(mols_container)
    # Assert that the same length of input and output
    assert len(fps) == len(mols_container)

    fpsize = params["fpSize"]
    assert len(fps[0]) == fpsize


@pytest.mark.parametrize("transformer_class", test_transformers)
def test_picklable(transformer_class):
    # Test the different transformers
    transformer = transformer_class()
    p = transformer.get_params()

    with tempfile.NamedTemporaryFile() as f:
        pickle.dump(transformer, f)
        f.seek(0)
        t2 = pickle.load(f)
        print(p)
        print(vars(transformer))
        print(vars(t2))
        assert transformer.get_params() == t2.get_params()


@pytest.mark.parametrize("transfomer", test_transformers)
def assert_transformer_set_params(transfomer, new_params, mols_list):
    default_params = transfomer().get_params()

    for key in new_params.keys():
        tr = transfomer()
        params = tr.get_params()
        params[key] = new_params[key]

        fps_default = tr.transform(mols_list)

        tr.set_params(**params)
        new_tr = transfomer(**params)
        fps_reset_params = tr.transform(mols_list)
        fps_init_new_params = new_tr.transform(mols_list)

        # Now fp_default should not be the same as fp_reset_params

        assert ~np.all(
            [
                np.array_equal(fp_default, fp_reset_params)
                for fp_default, fp_reset_params in zip(fps_default, fps_reset_params)
            ]
        ), f"Assertation error, FP appears the same, although the {key} should be changed from {default_params[key]} to {params[key]}"
        # fp_reset_params and fp_init_new_params should however be the same
        assert np.all(
            [
                np.array_equal(fp_init_new_params, fp_reset_params)
                for fp_init_new_params, fp_reset_params in zip(
                    fps_init_new_params, fps_reset_params
                )
            ]
        ), f"Assertation error, FP appears to be different, although the {key} should be changed back as well as initialized to {params[key]}"


def test_morgan_set_params(chiral_mols_list):
    new_params = {
        "fpSize": 1024,
        "radius": 1,
        "useBondTypes": False,  # TODO, why doesn't this change the FP?
        "useChirality": True,
        "useCounts": True,
        "useFeatures": True,
    }

    assert_transformer_set_params(
        MorganFingerprintTransformer, new_params, chiral_mols_list
    )


def test_atompairs_set_params(chiral_mols_list):
    new_params = {
        #'atomInvariants': 1,
        #'confId': -1,
        #'fromAtoms': 1,
        #'ignoreAtoms': 0,
        "includeChirality": True,
        "maxLength": 3,
        "minLength": 3,
        "fpSize": 1024,
        #'nBitsPerEntry': 3, #TODO: seem deprecated with the generators?
        #'use2D': True, #TODO, understand why this can't be set different
        "useCounts": True,
    }

    assert_transformer_set_params(
        AtomPairFingerprintTransformer, new_params, chiral_mols_list
    )


def test_topologicaltorsion_set_params(chiral_mols_list):
    new_params = {  #'atomInvariants': 0,
        #'fromAtoms': 0,
        #'ignoreAtoms': 0,
        #'includeChirality': True, #TODO, figure out why this setting seems to give same FP wheter toggled or not
        "fpSize": 1024,
        #'nBitsPerEntry': 3, #Todo: not setable with the generators?
        "targetSize": 5,
        "useCounts": True,
    }

    assert_transformer_set_params(
        TopologicalTorsionFingerprintTransformer, new_params, chiral_mols_list
    )


def test_RDKitFPTransformer(chiral_mols_list):
    new_params = {  #'atomInvariantsGenerator': None,
        #'branchedPaths': False,
        #'countBounds': 0, #TODO: What does this do?
        "countSimulation": True,
        "fpSize": 1024,
        "maxPath": 3,
        "minPath": 2,
        "numBitsPerFeature": 3,
        "useBondOrder": False,  # TODO, why doesn't this change the FP?
        #'useHs': False, #TODO, why doesn't this change the FP?
    }
    assert_transformer_set_params(
        RDKitFingerprintTransformer, new_params, chiral_mols_list
    )
