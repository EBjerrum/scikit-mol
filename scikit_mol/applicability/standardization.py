"""
Standardization approach applicability domain.

This module was adapted from MLChemAD (https://github.com/OlivierBeq/MLChemAD)
Original work Copyright (c) 2023 Olivier J. M. Béquignon (MIT License)
Modifications Copyright (c) 2025 scikit-mol contributors (LGPL License)
See LICENSE.MIT in this directory for the original MIT license.
"""

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted


class StandardizationApplicabilityDomain(BaseEstimator, TransformerMixin):
    """Applicability domain based on standardized feature values.

    Samples are considered within the domain if their standardized features
    have a mean + z * std <= threshold, or if their maximum standardized
    value <= threshold, where z corresponds to the specified percentile
    assuming a normal distribution.

    Parameters
    ----------
    percentile : float, default=95.0
        Percentile for the confidence interval (0-100).
        Default 95.0 corresponds to ~2 standard deviations.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    scaler_ : StandardScaler
        Fitted standard scaler.
    threshold_ : float
        Current threshold for standardized values.

    Examples
    --------
    >>> from scikit_mol.applicability import StandardizationApplicabilityDomain
    >>> ad = StandardizationApplicabilityDomain(percentile=95)
    >>> ad.fit(X_train)
    >>> # Optionally adjust threshold using validation set
    >>> ad.fit_threshold(X_val, target_percentile=95)
    >>> predictions = ad.predict(X_test)

    References
    ----------
    .. [1] Roy, K., Kar, S., & Ambure, P. (2015). On a simple approach for
           determining applicability domain of QSAR models. Chemometrics and
           Intelligent Laboratory Systems, 145, 22-29.
    """

    def __init__(self, percentile=95.0):
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must be between 0 and 100")
        self.percentile = percentile

    def fit(self, X, y=None):
        """Fit the standardization applicability domain.

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

        self.scaler_ = StandardScaler()
        self.scaler_.fit(X)

        # Convert percentile to z-score for initial threshold
        self.threshold_ = stats.norm.ppf(self.percentile / 100)

        return self

    def fit_threshold(self, X, target_percentile=95):
        """Update the threshold using new data without refitting the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to compute threshold from.
        target_percentile : float, default=95
            Target percentile of samples to include within domain.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        check_is_fitted(self)
        X = check_array(X)

        if not 0 <= target_percentile <= 100:
            raise ValueError("target_percentile must be between 0 and 100")

        # Calculate scores for the provided data
        scores = self.transform(X).ravel()

        # Set threshold to achieve desired percentile
        self.threshold_ = np.percentile(scores, target_percentile)

        return self

    def transform(self, X):
        """Calculate standardized feature statistics for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        scores : ndarray of shape (n_samples, 1)
            The maximum of:
            1. Maximum absolute standardized value
            2. Mean + z * std of standardized values
            where z corresponds to the specified percentile.
            Higher values indicate samples further from the training data.
        """
        check_is_fitted(self)
        X = check_array(X)

        # Standardize features
        X_std = self.scaler_.transform(X)

        # Calculate statistics
        max_vals = np.max(np.abs(X_std), axis=1)
        means = np.mean(X_std, axis=1)
        stds = np.std(X_std, axis=1)
        z_score = -stats.norm.ppf(self.percentile / 100)  # negative for lower tail
        mean_std = means + z_score * stds

        # Return maximum of the two criteria
        scores = np.maximum(max_vals, mean_std)
        return scores.reshape(-1, 1)

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
        return np.where(scores <= self.threshold_, 1, -1)
