"""Tests specific to Bounding Box applicability domain."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scikit_mol.applicability import BoundingBoxApplicabilityDomain


def test_bounding_box_bounds(ad_test_data):
    """Test the bounds calculation."""
    X_train, _, _ = ad_test_data
    ad = BoundingBoxApplicabilityDomain(percentile=(1, 99))
    ad.fit(X_train)

    # Check bounds match numpy percentile
    expected_min = np.percentile(X_train, 1, axis=0)
    expected_max = np.percentile(X_train, 99, axis=0)

    assert np.allclose(ad.min_, expected_min)
    assert np.allclose(ad.max_, expected_max)


def test_bounding_box_violations():
    """Test violation counting."""
    X_train = np.array([[1, 1], [2, 2], [3, 3]])
    X_test = np.array(
        [
            [2, 2],  # Inside bounds (0 violations)
            [0, 2],  # One violation
            [0, 4],  # Two violations
        ]
    )

    ad = BoundingBoxApplicabilityDomain(percentile=(0, 100))
    ad.fit(X_train)

    scores = ad.transform(X_test)
    assert scores[0, 0] == 0  # Inside bounds
    assert scores[1, 0] == 1  # One violation
    assert scores[2, 0] == 2  # Two violations


def test_bounding_box_percentile_validation():
    """Test percentile parameter validation."""
    # Invalid single percentile
    with pytest.raises(ValueError):
        BoundingBoxApplicabilityDomain(percentile=101)

    # Invalid tuple length
    with pytest.raises(ValueError):
        BoundingBoxApplicabilityDomain(percentile=(1, 2, 3))

    # Invalid order
    with pytest.raises(ValueError):
        BoundingBoxApplicabilityDomain(percentile=(99, 1))


def test_bounding_box_pipeline():
    """Test bounding box works in pipeline with scaling."""
    X = np.random.randn(10, 5)
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("ad", BoundingBoxApplicabilityDomain())]
    )

    # Should run without errors
    pipe.fit(X)
    scores = pipe.transform(X)
    assert scores.shape == (len(X), 1)
