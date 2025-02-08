"""
Leverage-based applicability domain.

This module was adapted from MLChemAD (https://github.com/OlivierBeq/MLChemAD) as described in the README.md file.
Original work Copyright (c) 2023 Olivier J. M. Béquignon (MIT License)
Modifications Copyright (c) 2025 scikit-mol contributors (LGPL License)
See LICENSE.MIT in this directory for the original MIT license.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class LeverageApplicabilityDomain(BaseEstimator, TransformerMixin):
    """Applicability domain defined using the leverage approach.

    The leverage approach measures how far a sample is from the center of the
    feature space using the diagonal elements of the hat matrix H = X(X'X)^(-1)X'.

    Parameters
    ----------
    threshold_factor : float, default=3
        Factor used in calculating the leverage threshold h* = threshold_factor * (p+1)/n
        where p is the number of features and n is the number of samples.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    threshold_ : float
        Calculated leverage threshold.
    var_covar_ : ndarray of shape (n_features, n_features)
        Variance-covariance matrix of the training data.
    """

    def __init__(self, threshold_factor=3):
        self.threshold_factor = threshold_factor

    def fit(self, X, y=None):
        """Fit the leverage applicability domain.

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
        n_samples = X.shape[0]

        # Calculate variance-covariance matrix
        self.var_covar_ = np.linalg.inv(X.T.dot(X))

        # Calculate threshold
        self.threshold_ = self.threshold_factor * (self.n_features_in_ + 1) / n_samples

        return self

    def transform(self, X):
        """Calculate leverage values for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        h : ndarray of shape (n_samples, 1)
            The leverage values. Higher values indicate samples further from
            the center of the training data.
        """
        check_is_fitted(self)
        X = check_array(X)

        # Calculate leverage values
        h = np.sum(X.dot(self.var_covar_) * X, axis=1)
        return h.reshape(-1, 1)

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
        return np.where(scores < self.threshold_, 1, -1)
