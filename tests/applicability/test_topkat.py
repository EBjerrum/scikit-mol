"""Tests for TopkatApplicabilityDomain."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from scikit_mol.applicability import TopkatApplicabilityDomain


@pytest.fixture
def ad_estimator():
    """Fixture providing a TopkatApplicabilityDomain instance."""
    return TopkatApplicabilityDomain()


def test_ops_transformation():
    """Test OPS transformation and distance calculation."""
    # Create simple test data
    X_train = np.array([[0, 0], [1, 1], [2, 2]])
    X_test = np.array([[0.5, 0.5], [10, 10]])

    # Fit AD model
    ad = TopkatApplicabilityDomain()
    ad.fit(X_train)

    # Check distances
    distances = ad.transform(X_test)
    assert distances.shape == (2, 1)
    assert distances[0] < distances[1]  # Interpolated point should have lower distance


def test_fixed_threshold():
    """Test that threshold is based on dimensionality."""
    X = np.random.randn(10, 3)
    ad = TopkatApplicabilityDomain()
    ad.fit(X)

    # Check threshold formula
    expected_threshold = 5 * (3 + 1) / (2 * 3)  # n_features = 3
    assert_array_almost_equal(ad.threshold_, expected_threshold)


def test_eigendecomposition():
    """Test eigendecomposition properties."""
    X = np.random.randn(10, 2)
    ad = TopkatApplicabilityDomain()
    ad.fit(X)

    # Check eigenvalue/vector shapes
    assert ad.eigen_val_.shape == (3,)  # n_features + 1
    assert ad.eigen_vec_.shape == (3, 3)  # (n_features + 1, n_features + 1)

    # Check eigenvalues are real and sorted
    assert np.all(np.isreal(ad.eigen_val_))
    assert np.all(np.diff(ad.eigen_val_) >= 0)  # Sorted in ascending order
