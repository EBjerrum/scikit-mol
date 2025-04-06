"""Tests specific to Isolation Forest applicability domain."""

import numpy as np

from scikit_mol.applicability import IsolationForestApplicabilityDomain


def test_refit_consistency():
    """Test consistency when refitting with same data."""
    X = np.random.RandomState(42).normal(0, 1, (100, 2))

    # Use fixed random state
    ad = IsolationForestApplicabilityDomain(
        n_estimators=100, contamination=0.1, random_state=42
    )

    # First fit
    ad.fit(X)
    scores1 = ad.transform(X)

    # Second fit
    ad.fit(X)
    scores2 = ad.transform(X)

    assert np.allclose(scores1, scores2)
