"""Tests specific to Convex Hull applicability domain."""

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scikit_mol.applicability import ConvexHullApplicabilityDomain


def test_convex_hull_simple():
    """Test with simple 2D data where result is obvious."""
    # Create a triangle of points
    X_train = np.array([[0, 0], [1, 0], [0, 1]])
    X_test = np.array(
        [
            [0.5, 0.25],  # Inside triangle
            [2, 2],  # Outside triangle
        ]
    )

    ad = ConvexHullApplicabilityDomain()
    ad.fit(X_train)

    scores = ad.transform(X_test)
    assert scores[0, 0] == 0.0  # Inside point
    assert scores[1, 0] == 1.0  # Outside point


def test_convex_hull_pipeline():
    """Test convex hull works in pipeline with dimensionality reduction."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2)),  # Reduce to 2D for speed
            ("ad", ConvexHullApplicabilityDomain()),
        ]
    )

    # Generate random high-dimensional data
    X = np.random.randn(10, 5)

    # Should run without errors
    pipe.fit(X)
    scores = pipe.transform(X)
    assert scores.shape == (len(X), 1)
    assert np.all((scores == 0) | (scores == 1))  # Binary output


def test_convex_hull_numerical_stability():
    """Test numerical stability with nearly colinear points."""
    X_train = np.array(
        [
            [0, 0],
            [1, 0],
            [2, 1e-10],  # Nearly colinear
        ]
    )
    X_test = np.array([[0.5, 0]])

    ad = ConvexHullApplicabilityDomain()
    ad.fit(X_train)

    # Should not raise and give consistent results
    scores = ad.transform(X_test)
    assert np.all(np.isfinite(scores))


def test_convex_hull_single_point():
    """Test behavior with single point (degenerate hull)."""
    X_train = np.array([[1, 1]])
    X_test = np.array([[1, 1], [2, 2]])

    ad = ConvexHullApplicabilityDomain()
    ad.fit(X_train)

    scores = ad.transform(X_test)
    assert scores[0, 0] == 0.0  # Same point
    assert scores[1, 0] == 1.0  # Different point
