"""Tests specific to Leverage applicability domain."""

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scikit_mol.applicability import LeverageApplicabilityDomain


def test_leverage_statistical_threshold(ad_test_data):
    """Test the statistical threshold calculation."""
    X_train, _, _ = ad_test_data
    ad = LeverageApplicabilityDomain(threshold_factor=3)
    ad.fit(X_train)

    # Check threshold matches formula h* = 3 * (p+1)/n
    n_samples, n_features = X_train.shape
    expected_threshold = 3 * (n_features + 1) / n_samples
    assert np.isclose(ad.threshold_, expected_threshold)


def test_leverage_pipeline(reduced_fingerprints):
    """Test leverage works in pipeline with scaling and PCA."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95)),
            ("ad", LeverageApplicabilityDomain()),
        ]
    )

    # Should run without errors
    pipe.fit(reduced_fingerprints)
    scores = pipe.transform(reduced_fingerprints)
    assert scores.shape == (len(reduced_fingerprints), 1)


def test_leverage_threshold_factor():
    """Test different threshold factors."""
    X = np.array([[1, 2], [3, 4], [5, 6]])

    ad1 = LeverageApplicabilityDomain(threshold_factor=3)
    ad2 = LeverageApplicabilityDomain(threshold_factor=2)

    ad1.fit(X)
    ad2.fit(X)

    # Higher threshold factor should result in higher threshold
    assert ad1.threshold_ > ad2.threshold_


def test_leverage_var_covar_matrix(ad_test_data):
    """Test the variance-covariance matrix calculation."""
    X_train, _, _ = ad_test_data
    ad = LeverageApplicabilityDomain()
    ad.fit(X_train)

    # Check matrix properties
    assert ad.var_covar_.shape == (X_train.shape[1], X_train.shape[1])
    assert np.allclose(ad.var_covar_, ad.var_covar_.T)  # Should be symmetric
