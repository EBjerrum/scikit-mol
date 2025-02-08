"""
Isolation Forest applicability domain.

This module was adapted from MLChemAD (https://github.com/OlivierBeq/MLChemAD)
Original work Copyright (c) 2023 Olivier J. M. Béquignon (MIT License)
Modifications Copyright (c) 2025 scikit-mol contributors (LGPL License)
See LICENSE.MIT in this directory for the original MIT license.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.utils.validation import check_array, check_is_fitted


class IsolationForestApplicabilityDomain(BaseEstimator, TransformerMixin):
    """Applicability domain based on Isolation Forest.

    Uses Isolation Forest to identify outliers based on the isolation depth
    of samples in random decision trees.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    contamination : float, default=0.01
        Expected proportion of outliers in the training data.
    random_state : int or RandomState, default=None
        Controls the randomness of the forest.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    iforest_ : IsolationForest
        Fitted isolation forest model.

    Examples
    --------
    >>> from scikit_mol.applicability import IsolationForestApplicabilityDomain
    >>> ad = IsolationForestApplicabilityDomain(contamination=0.1)
    >>> ad.fit(X_train)
    >>> predictions = ad.predict(X_test)

    References
    ----------
    .. [1] Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest.
           In 2008 Eighth IEEE International Conference on Data Mining (pp. 413-422).
    """

    def __init__(self, n_estimators=100, contamination=0.01, random_state=None):
        if not 0 < contamination < 1:
            raise ValueError("contamination must be between 0 and 1")

        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the isolation forest applicability domain.

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

        self.iforest_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.iforest_.fit(X)

        self.fit_threshold(X)

        return self

    def transform(self, X):
        """Calculate anomaly scores for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        scores : ndarray of shape (n_samples, 1)
            The anomaly scores of the samples.
            The lower the score, the more abnormal the sample.
        """
        check_is_fitted(self)
        X = check_array(X)

        scores = self.iforest_.score_samples(X)
        return scores.reshape(-1, 1)

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

        # Get decision function scores
        scores = self.iforest_.score_samples(X)

        # Set threshold to achieve desired percentile
        self.threshold_ = np.percentile(scores, 100 - target_percentile)

        return self

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
        if hasattr(self, "threshold_"):
            return np.where(scores > self.threshold_, 1, -1)
        return self.iforest_.predict(X)
