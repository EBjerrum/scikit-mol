"""Tests for EstimatorUnion adapter."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from scikit_mol.adapters import EstimatorUnion
from scikit_mol.applicability import (
    MahalanobisApplicabilityDomain,
    StandardizationApplicabilityDomain,
)

# Use existing fixtures
from .fixtures import (
    atompair_transformer,
    mols_list,
    morgan_transformer,
    skip_pandas_output_test,
)


def test_estimator_union_basic(morgan_transformer, atompair_transformer):
    """Test basic functionality of EstimatorUnion."""

    union = EstimatorUnion(
        [
            ("fp1", morgan_transformer),
            (
                "fp2",
                atompair_transformer,
            ),  # Different radius for different features
        ]
    )

    # Test unfitted raises exception
    with pytest.raises(NotFittedError):
        union.transform(mols_list)

    # Test fit and transform
    union.fit(mols_list)
    features = union.transform(mols_list)

    # Check output shape
    n_fp = morgan_transformer().fpSize
    assert features.shape == (len(mols_list), n_fp * 2)


def test_estimator_union_with_ad(morgan_transformer):
    """Test EstimatorUnion with applicability domain estimator."""
    union = EstimatorUnion(
        [
            ("fp", morgan_transformer),
            ("ad", MahalanobisApplicabilityDomain()),
        ],
        method_resolution_order=("transform", "transform_score"),
    )

    union.fit(mols_list)
    features = union.transform(mols_list)

    # Check output shape (fingerprints + 1 AD score)
    n_fp = morgan_transformer().fpSize
    assert features.shape == (len(mols_list), n_fp + 1)


def test_estimator_union_mixed_methods(morgan_transformer):
    """Test EstimatorUnion with different methods specified."""
    union = EstimatorUnion(
        [
            ("scale", StandardScaler(), "transform"),
            ("ad", StandardizationApplicabilityDomain(), "transform_score"),
            ("fp", morgan_transformer, "transform"),
        ]
    )

    # Create some numeric data for StandardScaler
    X = morgan_transformer.fit_transform(mols_list)
    union.fit(X)
    features = union.transform(X)

    # Check output shape
    assert features.shape[0] == len(X)
    assert features.shape[1] == X.shape[1] * 2 + 1  # scaled + fp + 1 AD score


@skip_pandas_output_test
def test_estimator_union_pandas_output(pandas_output, morgan_transformer):
    """Test pandas DataFrame output from EstimatorUnion."""
    union = EstimatorUnion(
        [
            ("fp", morgan_transformer),
            ("ad", MahalanobisApplicabilityDomain(), "transform_score"),
        ]
    )

    union.fit(mols_list)
    features = union.transform(mols_list)

    # Check output type and structure
    assert isinstance(features, pd.DataFrame)
    assert len(features) == len(mols_list)

    # Check column names
    fp_cols = [f"fp_{i}" for i in range(morgan_transformer.fpSize)]
    expected_cols = fp_cols + ["Mahalanobis"]
    assert features.columns.tolist() == expected_cols


def test_estimator_union_invalid_method(morgan_transformer):
    """Test EstimatorUnion with invalid method specification."""
    with pytest.raises(ValueError):
        EstimatorUnion([("fp", morgan_transformer, "invalid_method")])


def test_estimator_union_get_feature_names_out(morgan_transformer):
    """Test feature names output from EstimatorUnion."""
    union = EstimatorUnion(
        [
            ("fp", morgan_transformer),
            ("ad", MahalanobisApplicabilityDomain(), "transform_score"),
        ]
    )

    union.fit(mols_list)
    feature_names = union.get_feature_names_out()

    # Check number and format of feature names
    n_fp = morgan_transformer().fpSize
    assert len(feature_names) == n_fp + 1
    assert all(name.startswith("fp_") for name in feature_names[:-1])
    assert feature_names[-1] == "Mahalanobis"


def test_estimator_union_partial_fit(morgan_transformer):
    """Test EstimatorUnion with some estimators already fitted."""
    fp = morgan_transformer.fit(mols_list)
    ad = MahalanobisApplicabilityDomain()

    union = EstimatorUnion([("fp", fp), ("ad", ad, "transform_score")])

    # Should work since fp is already fitted
    features = union.fit_transform(mols_list)
    assert features.shape == (len(mols_list), fp.fpSize + 1)
