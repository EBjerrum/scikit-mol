"""
Convex hull applicability domain.

This module was adapted from MLChemAD (https://github.com/OlivierBeq/MLChemAD)
Original work Copyright (c) 2023 Olivier J. M. Béquignon (MIT License)
Modifications Copyright (c) 2025 scikit-mol contributors (LGPL License)
See LICENSE.MIT in this directory for the original MIT license.
"""

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

from .base import BaseApplicabilityDomain


class ConvexHullApplicabilityDomain(BaseApplicabilityDomain):
    """Applicability domain defined as the convex hull of the training data.

    The convex hull approach determines if a point belongs to the convex hull of the
    training set by checking if it can be represented as a convex combination of
    training points.

    Parameters
    ----------
    percentile : float or None, default=None
        Not used, present for API consistency.
    feature_prefix : str, default="ConvexHull"
        Prefix for feature names in output.

    Notes
    -----
    The method is based on the `highs` solver from `scipy.optimize`. Note that this
    method can be computationally expensive for high-dimensional data or large
    training sets, as it requires solving a linear programming problem for each
    test point.

    For high-dimensional data (e.g., fingerprints), consider using dimensionality
    reduction before applying this method.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    points_ : ndarray of shape (n_features + 1, n_samples)
        Transformed training points used for convex hull calculations.
    threshold_ : float
        Fixed at 0.5 since output is binary (inside/outside hull).
    """

    _scoring_convention = "high_outside"
    _supports_threshold_fitting = False

    def __init__(
        self, percentile: Optional[float] = None, feature_prefix: str = "ConvexHull"
    ) -> None:
        super().__init__(percentile=None, feature_prefix=feature_prefix)
        self.threshold_ = 0.5  # Fixed threshold since output is binary

    def fit(
        self, X: ArrayLike, y: Optional[Any] = None
    ) -> "ConvexHullApplicabilityDomain":
        """Fit the convex hull applicability domain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : ConvexHullApplicabilityDomain
            Returns the instance itself.
        """
        X = self._validate_data(X)
        self.n_features_in_ = X.shape[1]

        # Add ones column and transpose for convex hull calculations
        self.points_ = np.r_[X.T, np.ones((1, X.shape[0]))].astype(np.float32)

        return self

    def _transform(self, X: NDArray) -> NDArray[np.float64]:
        """Calculate distance from convex hull for each sample.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        distances : ndarray of shape (n_samples, 1)
            Distance from convex hull. Zero for points inside the hull,
            positive for points outside.
        """
        distances = []
        for sample in X:
            # Append 1 to sample vector
            sample_ext = np.r_[sample, 1].astype(np.float32)

            # Try to solve the linear programming problem
            result = optimize.linprog(
                np.ones(self.points_.shape[1], dtype=np.float32),
                A_eq=self.points_,
                b_eq=sample_ext,
                method="highs",
            )
            # Distance is positive if no solution found, 0 if solution exists
            distances.append(0.0 if result.success else 1.0)

        return np.array(distances).reshape(-1, 1)

    def predict(self, X):
        """Predict whether samples are within the applicability domain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Returns 1 for samples inside the domain and -1 for samples outside
            (following scikit-learn's convention for outlier detection).
        """
        scores = self._transform(X).ravel()
        return np.where(scores == 0, 1, -1)
