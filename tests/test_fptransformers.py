import pickle
import tempfile

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from sklearn import clone

from scikit_mol.fingerprints import (
    AvalonFingerprintTransformer,
    MACCSKeysFingerprintTransformer,
    MHFingerprintTransformer,
    SECFingerprintTransformer,
)

from .fixtures import (
    chiral_mols_list,
    chiral_smiles_list,
    fingerprint,
    mols_container,
    mols_list,
    mols_with_invalid_container,
    smiles_container,
    smiles_list,
    smiles_list_with_invalid,
)


@pytest.fixture
def maccs_transformer():
    return MACCSKeysFingerprintTransformer()


@pytest.fixture
def secfp_transformer():
    return SECFingerprintTransformer()


@pytest.fixture
def mhfp_transformer():
    return MHFingerprintTransformer()


@pytest.fixture
def avalon_transformer():
    return AvalonFingerprintTransformer()


def test_clonability(
    maccs_transformer,
    secfp_transformer,
    mhfp_transformer,
    avalon_transformer,
):
    for t in [
        maccs_transformer,
        secfp_transformer,
        mhfp_transformer,
        avalon_transformer,
    ]:
        params = t.get_params()
        t2 = clone(t)
        params_2 = t2.get_params()
        # Parameters of cloned transformers should be the same
        assert all([params[key] == params_2[key] for key in params.keys()])
        # Cloned transformers should not be the same object
        assert t2 != t


def test_set_params(
    secfp_transformer,
    mhfp_transformer,
    avalon_transformer,
):
    for t in [avalon_transformer]:
        params = t.get_params()
        # change extracted dictionary
        params["fpSize"] = 4242
        # change params in transformer
        t.set_params(fpSize=4242)
        # get parameters as dictionary and assert that it is the same
        params_2 = t.get_params()
        assert all([params[key] == params_2[key] for key in params.keys()])

    for t in [secfp_transformer, mhfp_transformer]:
        params = t.get_params()
        params["fpSize"] = 4242
        t.set_params(fpSize=4242)
        params_2 = t.get_params()
        assert all([params[key] == params_2[key] for key in params.keys()])


def test_transform(
    mols_container,
    maccs_transformer,
    secfp_transformer,
    mhfp_transformer,
    avalon_transformer,
):
    # Test the different transformers
    for t in [
        maccs_transformer,
        secfp_transformer,
        mhfp_transformer,
        avalon_transformer,
    ]:
        params = t.get_params()
        print(type(t), params)
        fps = t.transform(mols_container)
        # Assert that the same length of input and output
        assert len(fps) == len(mols_container)

        # assert that the size of the fingerprint is the expected size
        fpsize = params["fpSize"]

        assert len(fps[0]) == fpsize


def test_transform_parallel(
    mols_container,
    maccs_transformer,
    secfp_transformer,
    mhfp_transformer,
    avalon_transformer,
):
    # Test the different transformers
    for t in [
        maccs_transformer,
        secfp_transformer,
        mhfp_transformer,
        avalon_transformer,
    ]:
        t.set_params(n_jobs=2)
        params = t.get_params()
        fps = t.transform(mols_container)
        # Assert that the same length of input and output
        assert len(fps) == len(mols_container)

        # assert that the size of the fingerprint is the expected size
        fpsize = params["fpSize"]

        assert len(fps[0]) == fpsize


def test_picklable(
    maccs_transformer,
    secfp_transformer,
    avalon_transformer,
):
    # Test the different transformers
    for t in [
        maccs_transformer,
        secfp_transformer,
        avalon_transformer,
    ]:
        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(t, f)
            f.seek(0)
            t2 = pickle.load(f)
            assert t.get_params() == t2.get_params()


def assert_transformer_set_params(tr_class, new_params, mols_list):
    default_params = tr_class().get_params()
    for key in new_params.keys():
        tr = tr_class()
        params = tr.get_params()
        params[key] = new_params[key]

        fps_default = tr.transform(mols_list)

        tr.set_params(**params)
        new_tr = tr_class(**params)

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


def test_SECFingerprintTransformer(chiral_mols_list):
    new_params = {
        "isomeric": True,
        "kekulize": True,
        "fpSize": 1048,
        "min_radius": 2,
        #'n_permutations': 2, # The SECFp is not using this setting
        "radius": 2,
        "rings": False,
        #'seed': 1 # The SECFp is not using this setting
    }
    assert_transformer_set_params(
        SECFingerprintTransformer, new_params, chiral_mols_list
    )


def test_MHFingerprintTransformer(chiral_mols_list):
    new_params = {
        "radius": 2,
        "rings": False,
        "isomeric": True,
        "kekulize": True,
        "min_radius": 2,
        "fpSize": 4096,
        "seed": 44,
    }
    assert_transformer_set_params(
        MHFingerprintTransformer, new_params, chiral_mols_list
    )


def test_AvalonFingerprintTransformer(chiral_mols_list):
    new_params = {
        "fpSize": 1024,
        "isQuery": True,
        # 'resetVect': True, #TODO: this doesn't change the FP
        "bitFlags": 32767,
    }
    assert_transformer_set_params(
        AvalonFingerprintTransformer, new_params, chiral_mols_list
    )


def test_transform_with_safe_inference_mode(
    mols_with_invalid_container,
    maccs_transformer,
    secfp_transformer,
    avalon_transformer,
):
    for t in [
        maccs_transformer,
        secfp_transformer,
        avalon_transformer,
    ]:
        t.set_params(safe_inference_mode=True)
        print(type(t))
        fps = t.transform(mols_with_invalid_container)

        assert len(fps) == len(mols_with_invalid_container)

        # Check that the last row (corresponding to the InvalidMol) contains NaNs
        assert np.all(fps.mask[-1])

        # Check that other rows don't contain NaNs
        assert not np.any(fps.mask[:-1])


def test_transform_without_safe_inference_mode(
    mols_with_invalid_container,
    maccs_transformer,
    secfp_transformer,
    avalon_transformer,
    # MHFP seem to accept invalid mols and return 0,0,0,0's
):
    for t in [
        maccs_transformer,
        secfp_transformer,
        avalon_transformer,
    ]:
        t.set_params(safe_inference_mode=False)
        with pytest.raises(
            Exception
        ):  # You might want to be more specific about the exception type
            print(f"testing {type(t)}")
            t.transform(mols_with_invalid_container)


# Add this test to check parallel processing with error handling
def test_transform_parallel_with_safe_inference_mode(
    mols_with_invalid_container,
    maccs_transformer,
    secfp_transformer,
    avalon_transformer,
):
    for t in [
        maccs_transformer,
        secfp_transformer,
        avalon_transformer,
    ]:
        t.set_params(safe_inference_mode=True, n_jobs=2)
        fps = t.transform(mols_with_invalid_container)

        assert len(fps) == len(mols_with_invalid_container)

        print(fps.mask)
        # Check that the last row (corresponding to the InvalidMol) is masked
        assert np.all(
            fps.mask[-1]
        )  # Mask should be true for all elements in the last row

        # Check that other rows don't contain any masked values
        assert not np.any(fps.mask[:-1, :])
