"""Tests for LocalOutlierFactorApplicabilityDomain."""

import numpy as np
import pytest

from scikit_mol.applicability import LocalOutlierFactorApplicabilityDomain


@pytest.fixture
def ad_estimator():
    """Fixture providing a LocalOutlierFactorApplicabilityDomain instance."""
    return LocalOutlierFactorApplicabilityDomain()


def test_n_neighbors_effect():
    """Test effect of n_neighbors parameter on scores."""
    # Create data with clear outlier
    X = np.vstack([np.random.randn(50, 2), [[10, 10]]])
    outlier = np.array([[10, 10]])

    # Compare different n_neighbors settings
    ad_small = LocalOutlierFactorApplicabilityDomain(n_neighbors=2)
    ad_large = LocalOutlierFactorApplicabilityDomain(n_neighbors=5)

    ad_small.fit(X)
    ad_large.fit(X)

    score_small = ad_small.transform(outlier)
    score_large = ad_large.transform(outlier)

    # Scores should be different but both should identify the point as an outlier
    assert score_small != score_large
    assert ad_small.predict(outlier) == -1
    assert ad_large.predict(outlier) == -1


def test_metric_parameter():
    """Test different metric parameters."""
    metrics = ["euclidean", "manhattan", "cosine"]
    X = np.random.randn(10, 2)

    for metric in metrics:
        ad = LocalOutlierFactorApplicabilityDomain(metric=metric)
        ad.fit(X)
        scores = ad.transform(X)
        assert scores.shape == (10, 1)


def test_contamination_effect():
    """Test effect of contamination parameter on predictions."""
    X = np.random.randn(100, 2)

    # Compare different contamination levels
    ad_low = LocalOutlierFactorApplicabilityDomain(contamination=0.05)
    ad_high = LocalOutlierFactorApplicabilityDomain(contamination=0.25)

    ad_low.fit(X)
    ad_high.fit(X)

    pred_low = ad_low.predict(X)
    pred_high = ad_high.predict(X)

    # Higher contamination should result in more outliers
    assert np.sum(pred_high == -1) > np.sum(pred_low == -1)
