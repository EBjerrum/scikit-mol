"""Tests for KernelDensityApplicabilityDomain."""

import pytest

from scikit_mol.applicability import KernelDensityApplicabilityDomain


@pytest.fixture
def ad_estimator():
    """Fixture providing a KernelDensityApplicabilityDomain instance."""
    return KernelDensityApplicabilityDomain()


def test_kernel_parameter():
    """Test different kernel parameters."""
    kernels = ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]
    # Create data with clear density gradient
    X = [[0, 0], [0.1, 0.1], [0.2, 0.2], [2, 2]]

    for kernel in kernels:
        ad = KernelDensityApplicabilityDomain(kernel=kernel)
        ad.fit(X)
        scores = ad.transform(X)
        assert scores.shape == (4, 1)
        # First point should have higher density than last point
        assert scores[0, 0] > scores[-1, 0], f"Failed for kernel {kernel}"


def test_bandwidth_effect():
    """Test effect of bandwidth parameter on scores."""
    X = [[0, 0], [1, 1], [2, 2]]
    test_point = [[10, 10]]  # Far from training data

    # Larger bandwidth should give higher scores to outliers
    ad_small = KernelDensityApplicabilityDomain(bandwidth=0.1)
    ad_large = KernelDensityApplicabilityDomain(bandwidth=10.0)

    ad_small.fit(X)
    ad_large.fit(X)

    score_small = ad_small.transform(test_point)
    score_large = ad_large.transform(test_point)

    assert score_large[0, 0] > score_small[0, 0]
