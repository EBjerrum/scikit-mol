"""Tests specific to KNN applicability domain."""

import numpy as np
import pytest

from scikit_mol.applicability import KNNApplicabilityDomain
from scikit_mol.fingerprints import MorganFingerprintTransformer


@pytest.fixture
def binary_fingerprints(mols_list):
    """Binary fingerprints for testing Tanimoto distance."""
    return MorganFingerprintTransformer(fpSize=1024).fit_transform(mols_list)


def test_knn_tanimoto(binary_fingerprints):
    """Test KNN with Tanimoto distance on binary fingerprints."""
    ad = KNNApplicabilityDomain(n_neighbors=3, distance_metric="tanimoto")
    ad.fit(binary_fingerprints)
    scores = ad.transform(binary_fingerprints)
    assert scores.shape == (len(binary_fingerprints), 1)
    assert np.all((0 <= scores) & (scores <= 1))  # Tanimoto distances are [0,1]


# ... other KNN-specific tests ...
