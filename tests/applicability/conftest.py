import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scikit_mol.applicability import (
    BoundingBoxApplicabilityDomain,
    ConvexHullApplicabilityDomain,
    HotellingT2ApplicabilityDomain,
    IsolationForestApplicabilityDomain,
    KernelDensityApplicabilityDomain,
    KNNApplicabilityDomain,
    LeverageApplicabilityDomain,
    LocalOutlierFactorApplicabilityDomain,
    MahalanobisApplicabilityDomain,
    StandardizationApplicabilityDomain,
    TopkatApplicabilityDomain,
)
from scikit_mol.fingerprints import MorganFingerprintTransformer

from ..fixtures import mols_list


@pytest.fixture(
    params=[
        (KNNApplicabilityDomain, dict(n_neighbors=3)),
        (LeverageApplicabilityDomain, dict(threshold_factor=3)),
        (BoundingBoxApplicabilityDomain, dict(percentile=(1, 99))),
        (ConvexHullApplicabilityDomain, dict()),  # No special parameters needed
        (HotellingT2ApplicabilityDomain, dict(significance=0.05)),
        (
            IsolationForestApplicabilityDomain,
            dict(
                n_estimators=100,
                contamination=0.1,
                random_state=42,  # Add fixed random state
            ),
        ),
        (
            KernelDensityApplicabilityDomain,
            dict(bandwidth=1.0, kernel="gaussian"),
        ),
        (
            LocalOutlierFactorApplicabilityDomain,
            dict(
                n_neighbors=3, contamination=0.1
            ),  # Reduced from 20 to 3 for small test datasets
        ),
        (MahalanobisApplicabilityDomain, dict()),  # No special parameters needed
        (StandardizationApplicabilityDomain, dict()),  # No special parameters needed
        (TopkatApplicabilityDomain, dict()),  # No special parameters needed
    ]
)
def ad_estimator(request):
    """Fixture providing fresh AD estimator instances."""
    estimator_class, params = request.param
    return estimator_class(**params)


@pytest.fixture
def reduced_fingerprints(mols_list):
    """Create dimensionality-reduced fingerprints for AD testing."""
    # Generate larger fingerprints first
    fps = MorganFingerprintTransformer(fpSize=1024).fit_transform(mols_list)
    # Reduce dimensionality while preserving ~90% variance
    pca = PCA(n_components=0.9)
    return StandardScaler().fit_transform(pca.fit_transform(fps))


@pytest.fixture
def binary_fingerprints(mols_list):
    """Binary fingerprints for testing e.g. Tanimoto distance."""
    return MorganFingerprintTransformer(fpSize=1024).fit_transform(mols_list)


@pytest.fixture
def ad_test_data():
    """Simple 2D data with clear in/out domain regions."""
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    X_train = rng.uniform(0, 1, (20, 2))
    X_test_in = rng.uniform(0.25, 0.75, (5, 2))
    X_test_out = rng.uniform(2, 3, (5, 2))
    X_test = np.vstack([X_test_in, X_test_out])
    y_test = np.array([1] * 5 + [-1] * 5)
    return X_train, X_test, y_test
