"""
Convex hull applicability domain.

This module was adapted from MLChemAD (https://github.com/OlivierBeq/MLChemAD)
Original work Copyright (c) 2023 Olivier J. M. Béquignon (MIT License)
Modifications Copyright (c) 2025 scikit-mol contributors (LGPL License)
See LICENSE.MIT in this directory for the original MIT license.
"""

import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class ConvexHullApplicabilityDomain(BaseEstimator, TransformerMixin):
    """Applicability domain defined as the convex hull of the training data.

    The convex hull approach determines if a point belongs to the convex hull of the
    training set by checking if it can be represented as a convex combination of
    training points.

    The method is based on the `highs` solver from the `scipy.optimize` module, but is still
    slow at inference time.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    points_ : ndarray of shape (n_features + 1, n_samples)
        Transformed training points used for convex hull calculations.
    """

    def fit(self, X, y=None):
        """Fit the convex hull applicability domain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = check_array(X)
        self.n_features_in_ = X.shape[1]

        # Add ones column and transpose for convex hull calculations
        self.points_ = np.r_[X.T, np.ones((1, X.shape[0]))].astype(np.float32)

        return self

    def transform(self, X):
        """Calculate distance from convex hull for each sample.

        A distance of 0 indicates the sample lies within the convex hull.
        Positive values indicate distance outside the hull.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        distances : ndarray of shape (n_samples, 1)
            Distance from convex hull. Zero for points inside the hull,
            positive for points outside.
        """
        check_is_fitted(self)
        X = check_array(X)

        # Calculate distances
        if X.ndim == 1:
            X = X.reshape(1, -1)

        distances = []
        for sample in X:
            # Append 1 to sample vector
            sample_ext = np.r_[sample, 1].astype(np.float16)

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
        scores = self.transform(X).ravel()
        return np.where(scores == 0, 1, -1)
