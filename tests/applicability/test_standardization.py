"""Tests for StandardizationApplicabilityDomain."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from scikit_mol.applicability import StandardizationApplicabilityDomain


@pytest.fixture
def ad_estimator():
    """Fixture providing a StandardizationApplicabilityDomain instance."""
    return StandardizationApplicabilityDomain()


def test_statistical_threshold():
    """Test normal distribution based statistical threshold."""
    # Create standard normal data
    n_samples = 1000
    n_features = 3
    X = np.random.randn(n_samples, n_features)

    # Fit with statistical threshold
    ad = StandardizationApplicabilityDomain(percentile=None)
    ad.fit(X)

    # For standard normal data, ~95% should be within threshold
    predictions = ad.predict(X)
    inside_ratio = np.mean(predictions == 1)
    assert 0.93 <= inside_ratio <= 0.97  # Allow some variation


def test_standardization():
    """Test standardization of features."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    ad = StandardizationApplicabilityDomain()
    ad.fit(X)

    # Transform data
    X_std = ad.scaler_.transform(X)

    # Check standardization properties
    assert_array_almost_equal(np.mean(X_std, axis=0), [0, 0])
    assert_array_almost_equal(np.std(X_std, axis=0), [1, 1])


def test_max_absolute_score():
    """Test that scores are maximum absolute standardized values."""
    # Create data with known standardized values
    X = np.array([[0, 0], [1, 2], [3, -4]])
    ad = StandardizationApplicabilityDomain()
    ad.fit(X)

    # Create test point with one extreme standardized value
    X_test = np.array([[0, 10]])  # Second feature will be very large when standardized
    scores = ad.transform(X_test)

    # Score should be the maximum absolute standardized value
    X_std = ad.scaler_.transform(X_test)
    expected_score = np.max(np.abs(X_std))
    assert_array_almost_equal(scores, [[expected_score]])


def test_outlier_detection():
    """Test outlier detection on simple dataset."""
    # Create data with clear outlier
    X = np.vstack([np.random.randn(50, 2), [[10, 10]]])
    outlier = np.array([[10, 10]])

    ad = StandardizationApplicabilityDomain()
    ad.fit(X)

    # Outlier should have high score and be predicted outside
    outlier_score = ad.transform(outlier)
    assert outlier_score[0, 0] > ad.threshold_
    assert ad.predict(outlier) == -1
