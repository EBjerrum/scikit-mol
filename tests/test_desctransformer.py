import time

import joblib
import numpy as np
import numpy.ma as ma
import pandas as pd
import pytest
import sklearn
from fixtures import (
    mols_container,
    mols_list,
    mols_with_invalid_container,
    skip_pandas_output_test,
    smiles_container,
    smiles_list,
    smiles_list_with_invalid,
)
from packaging.version import Version
from rdkit.Chem import Descriptors
from sklearn import clone
from sklearn.pipeline import Pipeline

from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.core import SKLEARN_VERSION_PANDAS_OUT
from scikit_mol.descriptors import MolecularDescriptorTransformer


@pytest.fixture
def default_descriptor_transformer():
    return MolecularDescriptorTransformer()


@pytest.fixture
def selected_descriptor_transformer():
    return MolecularDescriptorTransformer(
        desc_list=["HeavyAtomCount", "FractionCSP3", "RingCount", "MolLogP", "MolWt"]
    )


def test_descriptor_transformer_clonability(default_descriptor_transformer):
    for t in [default_descriptor_transformer]:
        params = t.get_params()
        t2 = clone(t)
        params_2 = t2.get_params()
        # Parameters of cloned transformers should be the same
        assert all([params[key] == params_2[key] for key in params.keys()])
        # Cloned transformers should not be the same object
        assert t2 != t


def test_descriptor_transformer_set_params(default_descriptor_transformer):
    for t in [default_descriptor_transformer]:
        params = t.get_params()
        # change extracted dictionary
        params["desc_list"] = ["HeavyAtomCount", "FractionCSP3"]
        # change params in transformer
        t.set_params(desc_list=["HeavyAtomCount", "FractionCSP3"])
        # get parameters as dictionary and assert that it is the same
        params_2 = t.get_params()
        assert all([params[key] == params_2[key] for key in params.keys()])
        assert len(default_descriptor_transformer.selected_descriptors) == 2


def test_descriptor_transformer_available_descriptors(
    default_descriptor_transformer, selected_descriptor_transformer
):
    # Default have as many as in RDkit and all are selected
    assert len(default_descriptor_transformer.available_descriptors) == len(
        Descriptors._descList
    )
    assert len(default_descriptor_transformer.selected_descriptors) == len(
        Descriptors._descList
    )
    # Default have as many as in RDkit but only 5 are selected
    assert len(selected_descriptor_transformer.available_descriptors) == len(
        Descriptors._descList
    )
    assert len(selected_descriptor_transformer.selected_descriptors) == 5


def test_descriptor_transformer_transform(
    mols_container, default_descriptor_transformer
):
    features = default_descriptor_transformer.transform(mols_container)
    assert len(features) == len(mols_container)
    assert len(features[0]) == len(Descriptors._descList)


def test_descriptor_transformer_wrong_descriptors():
    with pytest.raises(AssertionError):
        MolecularDescriptorTransformer(
            desc_list=[
                "Color",
                "Icecream content",
                "ChokolateDarkness",
                "Content42",
                "MolWt",
            ]
        )


def test_descriptor_transformer_parallel(mols_list, default_descriptor_transformer):
    default_descriptor_transformer.set_params(parallel=True)
    features = default_descriptor_transformer.transform(mols_list)
    assert len(features) == len(mols_list)
    assert len(features[0]) == len(Descriptors._descList)
    # Now with Rdkit 2022.3 creating a second transformer and running it, froze the process
    transformer2 = MolecularDescriptorTransformer(
        **default_descriptor_transformer.get_params()
    )
    features2 = transformer2.transform(mols_list)
    assert len(features2) == len(mols_list)
    assert len(features2[0]) == len(Descriptors._descList)


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


def test_transform_with_safe_inference_mode(mols_with_invalid_container):
    transformer = MolecularDescriptorTransformer(safe_inference_mode=True)
    descriptors = transformer.transform(mols_with_invalid_container)

    assert isinstance(descriptors, ma.MaskedArray)
    assert len(descriptors) == len(mols_with_invalid_container)

    # Check that the last row (corresponding to the InvalidMol) is fully masked
    assert np.all(descriptors.mask[-1])

    # Check that other rows are not masked
    assert not np.any(descriptors.mask[:-1])


def test_transform_without_safe_inference_mode(mols_with_invalid_container):
    transformer = MolecularDescriptorTransformer(safe_inference_mode=False)
    with pytest.raises(
        Exception
    ):  # You might want to be more specific about the exception type
        transformer.transform(mols_with_invalid_container)


def test_transform_parallel_with_safe_inference_mode(mols_with_invalid_container):
    transformer = MolecularDescriptorTransformer(
        safe_inference_mode=True, parallel=True
    )
    descriptors = transformer.transform(mols_with_invalid_container)

    assert isinstance(descriptors, ma.MaskedArray)
    assert len(descriptors) == len(mols_with_invalid_container)

    # Check that the last row (corresponding to the InvalidMol) is fully masked
    assert np.all(descriptors.mask[-1])

    # Check that other rows are not masked
    assert not np.any(descriptors.mask[:-1])


def test_transform_parallel_without_safe_inference_mode(mols_with_invalid_container):
    transformer = MolecularDescriptorTransformer(
        safe_inference_mode=False, parallel=True
    )
    with pytest.raises(
        Exception
    ):  # You might want to be more specific about the exception type
        transformer.transform(mols_with_invalid_container)


def test_safe_inference_mode_setting():
    transformer = MolecularDescriptorTransformer()
    assert not transformer.safe_inference_mode  # Default should be False

    transformer.set_params(safe_inference_mode=True)
    assert transformer.safe_inference_mode

    transformer.set_params(safe_inference_mode=False)
    assert not transformer.safe_inference_mode


# TODO, if these tests are run before the others, these tests will fail, probably due to pandas output?
@skip_pandas_output_test
def test_descriptor_transformer_pandas_output(
    mols_container,
    default_descriptor_transformer,
    selected_descriptor_transformer,
    pandas_output,
):
    for transformer in [
        default_descriptor_transformer,
        selected_descriptor_transformer,
    ]:
        features = transformer.transform(mols_container)
        assert isinstance(features, pd.DataFrame)
        assert features.shape[0] == len(mols_container)
        assert features.columns.tolist() == transformer.selected_descriptors


@skip_pandas_output_test
def test_descriptor_transformer_pandas_output_pipeline(
    smiles_container, default_descriptor_transformer, pandas_output
):
    pipeline = Pipeline(
        [("s2m", SmilesToMolTransformer()), ("desc", default_descriptor_transformer)]
    )
    features = pipeline.fit_transform(smiles_container)
    assert isinstance(features, pd.DataFrame)
    assert features.shape[0] == len(smiles_container)
    assert (
        features.columns.tolist() == default_descriptor_transformer.selected_descriptors
    )
