"""
Hotelling T² applicability domain.

This module was adapted from MLChemAD (https://github.com/OlivierBeq/MLChemAD)
Original work Copyright (c) 2023 Olivier J. M. Béquignon (MIT License)
Modifications Copyright (c) 2025 scikit-mol contributors (LGPL License)
See LICENSE.MIT in this directory for the original MIT license.
"""

import numpy as np
from scipy.stats import f as f_dist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class HotellingT2ApplicabilityDomain(BaseEstimator, TransformerMixin):
    """Applicability domain based on Hotelling's T² statistic.

    Uses Hotelling's T² statistic to define an elliptical confidence region
    around the training data. The threshold can be set using either the
    F-distribution (statistical approach) or adjusted using a validation set.

    Lower volume protrusion scores indicate samples closer to the training
    data center. By default, the threshold is set using the F-distribution
    with a significance level of 0.05 (95% confidence). When using fit_threshold,
    a target_percentile of 95 means that 95% of the validation samples with
    the lowest protrusion scores will be considered inside the domain.

    Parameters
    ----------
    significance : float, default=0.05
        Significance level for F-distribution threshold.
        Only used if fit_threshold is not called.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    t2_ : ndarray of shape (n_features,)
        Hotelling T² ellipse parameters.
    threshold_ : float
        Current threshold for volume protrusions.

    Examples
    --------
    >>> from scikit_mol.applicability import HotellingT2ApplicabilityDomain
    >>> ad = HotellingT2ApplicabilityDomain()
    >>> # Using F-distribution threshold (default)
    >>> ad.fit(X_train)
    >>> predictions = ad.predict(X_test)
    >>>
    >>> # Adjusting threshold using validation set
    >>> ad.fit_threshold(X_val, target_percentile=95)
    >>> predictions = ad.predict(X_test)

    References
    ----------
    .. [1] Hotelling, H. (1931). The generalization of Student's ratio.
           The Annals of Mathematical Statistics, 2(3), 360-378.
    """

    def __init__(self, significance=0.05):
        if not 0 < significance < 1:
            raise ValueError("significance must be between 0 and 1")
        self.significance = significance

    def fit(self, X, y=None):
        """Fit the Hotelling T² applicability domain.

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

        # Determine the Hotelling T² ellipse
        self.t2_ = np.sqrt((1 / n_samples) * (X**2).sum(axis=0))

        # Set initial threshold using F-distribution
        f_stat = (
            (n_samples - 1)
            / n_samples
            * self.n_features_in_
            * (n_samples**2 - 1)
            / (n_samples * (n_samples - self.n_features_in_))
        )
        f_stat *= f_dist.ppf(
            1 - self.significance, self.n_features_in_, n_samples - self.n_features_in_
        )
        self.threshold_ = f_stat

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

        # Calculate volume protrusions for validation set
        scores = self.transform(X).ravel()

        # Set threshold to achieve desired percentile (lower scores = inside domain)
        self.threshold_ = np.percentile(scores, 100 - target_percentile)

        return self

    def transform(self, X):
        """Calculate volume protrusion scores for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        scores : ndarray of shape (n_samples, 1)
            The volume protrusion scores. Higher values indicate samples
            further from the training data center.
        """
        check_is_fitted(self)
        X = check_array(X)

        # Calculate volume protrusions
        protrusions = (X**2 / self.t2_**2).sum(axis=1)
        return protrusions.reshape(-1, 1)

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
