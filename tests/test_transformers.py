# checking that the new transformers can work within a scikitlearn pipeline of the kind
# Pipeline([("s2m", SmilesToMol()), ("FP", FPTransformer()), ("RF", RandomForestRegressor())])
# using some test data stored in ./data/SLC6A4_active_excape_subset.csv

# to run as
# pytest tests/test_transformers.py --> tests/test_transformers.py::test_transformer PASSED


import pytest
import pandas as pd
from packaging.version import Version
import sklearn
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.core import SKLEARN_VERSION_PANDAS_OUT
from scikit_mol.fingerprints import (
    MACCSKeysFingerprintTransformer,
    RDKitFingerprintTransformer,
    AtomPairFingerprintTransformer,
    TopologicalTorsionFingerprintTransformer,
    MorganFingerprintTransformer,
    SECFingerprintTransformer,
    MHFingerprintTransformer,
    AvalonFingerprintTransformer,
)
from scikit_mol.fingerprints.baseclasses import BaseFpsTransformer

from scikit_mol.descriptors import MolecularDescriptorTransformer

from fixtures import (
    SLC6A4_subset,
    SLC6A4_subset_with_cddd,
    skip_pandas_output_test,
    mols_container,
    featurizer,
    combined_transformer,
)


def test_transformer(SLC6A4_subset):
    # load some toy data for quick testing on a small number of samples
    X_smiles, Y = SLC6A4_subset.SMILES, SLC6A4_subset.pXC50
    X_smiles = X_smiles.to_frame()
    X_train, X_test = X_smiles[:128], X_smiles[128:]
    Y_train, Y_test = Y[:128], Y[128:]

    # run FP with default parameters except when useCounts can be given as an argument
    FP_dict = {
        "MACCSTransformer": [MACCSKeysFingerprintTransformer, None],
        "RDKitFPTransformer": [RDKitFingerprintTransformer, None],
        "AtomPairFingerprintTransformer": [AtomPairFingerprintTransformer, False],
        "AtomPairFingerprintTransformer useCounts": [
            AtomPairFingerprintTransformer,
            True,
        ],
        "TopologicalTorsionFingerprintTransformer": [
            TopologicalTorsionFingerprintTransformer,
            False,
        ],
        "TopologicalTorsionFingerprintTransformer useCounts": [
            TopologicalTorsionFingerprintTransformer,
            True,
        ],
        "MorganTransformer": [MorganFingerprintTransformer, False],
        "MorganTransformer useCounts": [MorganFingerprintTransformer, True],
        "SECFingerprintTransformer": [SECFingerprintTransformer, None],
        "MHFingerprintTransformer": [MHFingerprintTransformer, None],
        "AvalonFingerprintTransformer": [AvalonFingerprintTransformer, None],
    }

    # fit on toy data and print train/test score if successful or collect the failed FP
    failed_FP = []
    for FP_name, (FP, useCounts) in FP_dict.items():
        try:
            print(
                f"\nrunning pipeline fitting and scoring for {FP_name} with useCounts={useCounts}"
            )
            if useCounts is None:
                pipeline = Pipeline(
                    [
                        ("s2m", SmilesToMolTransformer()),
                        ("FP", FP()),
                        ("RF", RandomForestRegressor()),
                    ]
                )
            else:
                pipeline = Pipeline(
                    [
                        ("s2m", SmilesToMolTransformer()),
                        ("FP", FP(useCounts=useCounts)),
                        ("RF", RandomForestRegressor()),
                    ]
                )
            pipeline.fit(X_train, Y_train)
            train_score = pipeline.score(X_train, Y_train)
            test_score = pipeline.score(X_test, Y_test)
            print(
                f"\nfitting and scoring completed train_score={train_score}, test_score={test_score}"
            )
        except:
            print(
                f"\n!!!! FAILED pipeline fitting and scoring for {FP_name} with useCounts={useCounts}"
            )
            failed_FP.append(FP_name)
            pass

    # overall result
    assert len(failed_FP) == 0, f"the following FP have failed {failed_FP}"


@skip_pandas_output_test
def test_transformer_pandas_output(SLC6A4_subset, pandas_output):
    # load some toy data for quick testing on a small number of samples
    X_smiles = SLC6A4_subset.SMILES
    X_smiles = X_smiles.to_frame()

    # run FP with default parameters except when useCounts can be given as an argument
    FP_dict = {
        "MACCSTransformer": [MACCSKeysFingerprintTransformer, None],
        "RDKitFPTransformer": [RDKitFingerprintTransformer, None],
        "AtomPairFingerprintTransformer": [AtomPairFingerprintTransformer, False],
        "AtomPairFingerprintTransformer useCounts": [
            AtomPairFingerprintTransformer,
            True,
        ],
        "TopologicalTorsionFingerprintTransformer": [
            TopologicalTorsionFingerprintTransformer,
            False,
        ],
        "TopologicalTorsionFingerprintTransformer useCounts": [
            TopologicalTorsionFingerprintTransformer,
            True,
        ],
        "MorganTransformer": [MorganFingerprintTransformer, False],
        "MorganTransformer useCounts": [MorganFingerprintTransformer, True],
        "SECFingerprintTransformer": [SECFingerprintTransformer, None],
        "MHFingerprintTransformer": [MHFingerprintTransformer, None],
        "AvalonFingerprintTransformer": [AvalonFingerprintTransformer, None],
    }

    # fit on toy data and check that the output is a pandas dataframe
    failed_FP = []
    for FP_name, (FP, useCounts) in FP_dict.items():
        try:
            print(
                f"\nrunning pipeline fitting and scoring for {FP_name} with useCounts={useCounts}"
            )
            if useCounts is None:
                pipeline = Pipeline([("s2m", SmilesToMolTransformer()), ("FP", FP())])
            else:
                pipeline = Pipeline(
                    [("s2m", SmilesToMolTransformer()), ("FP", FP(useCounts=useCounts))]
                )
            pipeline.fit(X_smiles)
            X_transformed = pipeline.transform(X_smiles)
            assert isinstance(
                X_transformed, pd.DataFrame
            ), f"the output of {FP_name} is not a pandas dataframe"
            assert (
                X_transformed.shape[0] == len(X_smiles)
            ), f"the number of rows in the output of {FP_name} is not equal to the number of samples"
            assert (
                len(X_transformed.columns) == pipeline.named_steps["FP"].fpSize
            ), f"the number of columns in the output of {FP_name} is not equal to the number of bits"
            print(f"\nfitting and transforming completed")

        except Exception as err:
            print(
                f"\n!!!! FAILED pipeline fitting and transforming for {FP_name} with useCounts={useCounts}"
            )
            print("\n".join(err.args))
            failed_FP.append(FP_name)
            pass

    # overall result
    assert (
        len(failed_FP) == 0
    ), f"the following FP have failed pandas transformation {failed_FP}"


@skip_pandas_output_test
def test_pandas_out_same_values(featurizer, mols_container):
    featurizer_default = sklearn.base.clone(featurizer)
    featurizer_default.set_output(transform="default")
    featurizer_pandas = sklearn.base.clone(featurizer)
    featurizer_pandas.set_output(transform="pandas")
    result_default = featurizer_default.fit_transform(mols_container)
    result_pandas = featurizer_pandas.fit_transform(mols_container)
    assert isinstance(result_default, np.ndarray)
    assert isinstance(result_pandas, pd.DataFrame)
    assert result_default.shape == result_pandas.shape
    featurizer_class_with_nan = MolecularDescriptorTransformer
    if isinstance(featurizer, featurizer_class_with_nan):
        assert (
            pd.isna(result_default) == pd.isna(result_pandas.values)
        ).all(), (
            "NaN values are not in the same positions in the default and pandas output"
        )
        nan_replacement = 0.0
        result_default = np.nan_to_num(result_default, nan=nan_replacement)
        result_pandas = result_pandas.fillna(nan_replacement)
    else:
        assert (result_default == result_pandas.values).all()


@skip_pandas_output_test
def test_combined_transformer_pandas_out(
    combined_transformer, SLC6A4_subset_with_cddd, pandas_output
):
    result = combined_transformer.fit_transform(SLC6A4_subset_with_cddd)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == SLC6A4_subset_with_cddd.shape[0]
    n_cddd_features = SLC6A4_subset_with_cddd.columns.str.match(r"^cddd_\d+$").sum()
    pipeline_skmol = combined_transformer.named_transformers_["pipeline-1"]
    featurizer_skmol = pipeline_skmol[-1]
    if isinstance(featurizer_skmol, BaseFpsTransformer):
        n_skmol_features = featurizer_skmol.fpSize
    elif isinstance(featurizer_skmol, MolecularDescriptorTransformer):
        n_skmol_features = len(featurizer_skmol.desc_list)
    else:
        raise ValueError(f"Unexpected featurizer type {type(featurizer_skmol)}")
    expected_n_features = n_cddd_features + n_skmol_features
    assert result.shape[1] == expected_n_features
