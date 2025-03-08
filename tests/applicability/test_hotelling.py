"""Tests specific to Hotelling T² applicability domain."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scikit_mol.applicability import HotellingT2ApplicabilityDomain


def test_hotelling_threshold():
    """Test F-distribution threshold calculation."""
    X = np.random.randn(100, 3)  # 100 samples, 3 features

    ad = HotellingT2ApplicabilityDomain(significance=0.05)
    ad.fit(X)

    # Threshold should be positive
    assert ad.threshold_ > 0

    # More stringent significance should give higher threshold
    ad_strict = HotellingT2ApplicabilityDomain(significance=0.01)
    ad_strict.fit(X)
    assert ad_strict.threshold_ > ad.threshold_


def test_hotelling_scores():
    """Test score calculation with known data."""
    # Create data with known center and spread
    X_train = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])

    X_test = np.array(
        [
            [0, 0],  # Center point
            [2, 0],  # Further out
            [10, 10],  # Far out
        ]
    )

    ad = HotellingT2ApplicabilityDomain()
    ad.fit(X_train)

    scores = ad.transform(X_test)

    # Scores should increase with distance from center
    assert scores[0, 0] < scores[1, 0] < scores[2, 0]


def test_hotelling_significance_validation():
    """Test significance parameter validation."""
    with pytest.raises(ValueError):
        HotellingT2ApplicabilityDomain(significance=0)

    with pytest.raises(ValueError):
        HotellingT2ApplicabilityDomain(significance=1)

    with pytest.raises(ValueError):
        HotellingT2ApplicabilityDomain(significance=-0.5)


def test_hotelling_pipeline():
    """Test Hotelling works in pipeline with scaling."""
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("ad", HotellingT2ApplicabilityDomain())]
    )

    X = np.random.randn(10, 5)

    # Should run without errors
    pipe.fit(X)
    scores = pipe.transform(X)
    assert scores.shape == (len(X), 1)
    assert np.all(scores >= 0)  # Scores should be non-negative


def test_hotelling_threshold_fitting():
    """Test threshold fitting with percentile."""
    X = np.random.randn(100, 3)

    ad = HotellingT2ApplicabilityDomain(percentile=90)
    ad.fit(X)

    # Get scores and check threshold matches 90th percentile
    scores = ad.transform(X)
    expected_threshold = np.percentile(scores, 90)
    assert np.isclose(ad.threshold_, expected_threshold)
