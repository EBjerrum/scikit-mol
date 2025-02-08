"""
Mahalanobis distance applicability domain.
"""

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class MahalanobisApplicabilityDomain(BaseEstimator, TransformerMixin):
    """Applicability domain based on Mahalanobis distance.

    Uses Mahalanobis distance to measure how many standard deviations a sample
    is from the training set mean, taking into account the covariance structure
    of the data. For multivariate normal data, the squared Mahalanobis distances
    follow a chi-square distribution.

    Parameters
    ----------
    percentile : float, default=95.0
        Percentile for the confidence region (0-100).
        Default 95.0 corresponds to ~2 standard deviations.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    mean_ : ndarray of shape (n_features,)
        Mean of training data.
    covariance_ : ndarray of shape (n_features, n_features)
        Covariance matrix of training data.
    inv_covariance_ : ndarray of shape (n_features, n_features)
        Inverse covariance matrix.
    threshold_ : float
        Current threshold for Mahalanobis distances.

    Examples
    --------
    >>> from scikit_mol.applicability import MahalanobisApplicabilityDomain
    >>> ad = MahalanobisApplicabilityDomain(percentile=95)
    >>> ad.fit(X_train)
    >>> # Using chi-square threshold (default)
    >>> predictions = ad.predict(X_test)
    >>>
    >>> # Adjusting threshold using validation set
    >>> ad.fit_threshold(X_val, target_percentile=95)
    >>> predictions = ad.predict(X_test)

    References
    ----------
    .. [1] De Maesschalck, R., Jouan-Rimbaud, D., & Massart, D. L. (2000).
           The Mahalanobis distance. Chemometrics and intelligent laboratory
           systems, 50(1), 1-18.
    """

    def __init__(self, percentile=95.0):
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must be between 0 and 100")
        self.percentile = percentile

    def fit(self, X, y=None):
        """Fit the Mahalanobis distance applicability domain.

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

        # Compute mean and covariance
        self.mean_ = np.mean(X, axis=0)
        self.covariance_ = np.cov(X, rowvar=False)
        self.inv_covariance_ = np.linalg.inv(self.covariance_)

        # Set initial threshold using chi-square distribution
        self.threshold_ = stats.chi2.ppf(self.percentile / 100, df=self.n_features_in_)

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

        # Calculate distances for validation set
        scores = self.transform(X).ravel()

        # Set threshold to achieve desired percentile (lower distances = inside domain)
        self.threshold_ = np.percentile(scores, 100 - target_percentile)

        return self

    def transform(self, X):
        """Calculate Mahalanobis distances for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        distances : ndarray of shape (n_samples, 1)
            The Mahalanobis distances. Higher values indicate samples
            further from the training data center.
        """
        check_is_fitted(self)
        X = check_array(X)

        # Center the data
        X_centered = X - self.mean_

        # Calculate Mahalanobis distances
        distances = np.sum(X_centered @ self.inv_covariance_ * X_centered, axis=1)
        return distances.reshape(-1, 1)

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
