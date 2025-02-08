"""
Kernel Density applicability domain.

This module was adapted from MLChemAD (https://github.com/OlivierBeq/MLChemAD)
Original work Copyright (c) 2023 Olivier J. M. Béquignon (MIT License)
Modifications Copyright (c) 2025 scikit-mol contributors (LGPL License)
See LICENSE.MIT in this directory for the original MIT license.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KernelDensity
from sklearn.utils.validation import check_array, check_is_fitted


class KernelDensityApplicabilityDomain(BaseEstimator, TransformerMixin):
    """Applicability domain based on kernel density estimation.

    Uses kernel density estimation to model the distribution of the training data.
    Samples with density below a threshold (determined by percentile of training
    data densities) are considered outside the domain.

    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth of the kernel.
    kernel : str, default='gaussian'
        The kernel to use. Options: ['gaussian', 'tophat', 'epanechnikov',
        'exponential', 'linear', 'cosine'].
    percentile : float, default=1.0
        The percentile of training set densities to use as threshold.
        Must be between 0 and 100.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    kde_ : KernelDensity
        Fitted kernel density estimator.
    threshold_ : float
        Density threshold for domain membership.

    Examples
    --------
    >>> from scikit_mol.applicability import KernelDensityApplicabilityDomain
    >>> ad = KernelDensityApplicabilityDomain(bandwidth=1.0)
    >>> ad.fit(X_train)
    >>> predictions = ad.predict(X_test)
    """

    def __init__(self, bandwidth=1.0, kernel="gaussian", percentile=1.0):
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must be between 0 and 100")

        self.bandwidth = bandwidth
        self.kernel = kernel
        self.percentile = percentile

    def fit(self, X, y=None):
        """Fit the kernel density applicability domain.

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

        # Fit KDE
        self.kde_ = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
        self.kde_.fit(X)

        # Set initial threshold based on training data
        self.fit_threshold(X)

        return self

    def transform(self, X):
        """Calculate log density scores for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        scores : ndarray of shape (n_samples, 1)
            The log density scores of the samples. Higher scores indicate samples
            more similar to the training data.
        """
        check_is_fitted(self)
        X = check_array(X)

        scores = self.kde_.score_samples(X)
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
        return np.where(scores >= self.threshold_, 1, -1)

    def fit_threshold(self, X):
        """Update the threshold using new data without refitting the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to compute threshold from.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        check_is_fitted(self)
        X = check_array(X)

        # Calculate density threshold from provided data
        densities = self.kde_.score_samples(X)
        self.threshold_ = np.percentile(densities, self.percentile)

        return self
