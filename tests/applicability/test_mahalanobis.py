"""Tests for MahalanobisApplicabilityDomain."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from scikit_mol.applicability import MahalanobisApplicabilityDomain


@pytest.fixture
def ad_estimator():
    """Fixture providing a MahalanobisApplicabilityDomain instance."""
    return MahalanobisApplicabilityDomain()


def test_statistical_threshold():
    """Test chi-square based statistical threshold."""
    # Create multivariate normal data
    n_samples = 1000
    n_features = 3
    mean = np.zeros(n_features)
    cov = np.eye(n_features)
    X = np.random.multivariate_normal(mean, cov, n_samples)

    # Fit with statistical threshold
    ad = MahalanobisApplicabilityDomain(percentile=None)
    ad.fit(X)

    # For standard normal data, ~95% should be within threshold
    predictions = ad.predict(X)
    inside_ratio = np.mean(predictions == 1)
    assert 0.93 <= inside_ratio <= 0.97  # Allow some variation


def test_mean_covariance():
    """Test mean and covariance computation."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    ad = MahalanobisApplicabilityDomain()
    ad.fit(X)

    # Check mean computation
    expected_mean = np.array([3, 4])
    assert_array_almost_equal(ad.mean_, expected_mean)

    # Check covariance computation
    expected_cov = np.array([[4, 4], [4, 4]])
    assert_array_almost_equal(ad.covariance_, expected_cov)


def test_distance_properties():
    """Test properties of Mahalanobis distances."""
    # Create data with clear outlier
    X = np.vstack([np.random.randn(50, 2), [[10, 10]]])
    outlier = np.array([[10, 10]])

    ad = MahalanobisApplicabilityDomain()
    ad.fit(X)

    # Distance to mean should be zero
    mean_dist = ad.transform(ad.mean_.reshape(1, -1))
    assert_array_almost_equal(mean_dist, [[0]], decimal=10)

    # Outlier should have large distance and be predicted outside
    outlier_dist = ad.transform(outlier)
    assert outlier_dist[0, 0] > ad.threshold_
    assert ad.predict(outlier) == -1
