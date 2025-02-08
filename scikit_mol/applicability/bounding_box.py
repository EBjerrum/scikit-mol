"""
Bounding box applicability domain.

This module was adapted from MLChemAD (https://github.com/OlivierBeq/MLChemAD)
Original work Copyright (c) 2023 Olivier J. M. Béquignon (MIT License)
Modifications Copyright (c) 2025 scikit-mol contributors (LGPL License)
See LICENSE.MIT in this directory for the original MIT license.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class BoundingBoxApplicabilityDomain(BaseEstimator, TransformerMixin):
    """Applicability domain defined by feature value ranges.

    Samples falling outside the allowed range for any feature are considered
    outside the domain.

    Parameters
    ----------
    percentile : float or tuple of float, default=(0.1, 99.9)
        Percentile(s) of the training set distribution used to define
        the bounding box. If float, uses (percentile, 100-percentile).

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    min_ : ndarray of shape (n_features,)
        Minimum allowed value for each feature.
    max_ : ndarray of shape (n_features,)
        Maximum allowed value for each feature.

    Examples
    --------
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.decomposition import PCA
    >>> from scikit_mol.applicability import BoundingBoxApplicabilityDomain

    Basic usage:
    >>> ad = BoundingBoxApplicabilityDomain(percentile=1)
    >>> ad.fit(X_train)
    >>> predictions = ad.predict(X_test)

    With preprocessing:
    >>> pipe = make_pipeline(
    ...     StandardScaler(),
    ...     BoundingBoxApplicabilityDomain(percentile=1)
    ... )
    >>> pipe.fit(X_train)
    >>> predictions = pipe.predict(X_test)

    With PCA preprocessing:
    >>> pipe = make_pipeline(
    ...     StandardScaler(),
    ...     PCA(n_components=0.9),
    ...     BoundingBoxApplicabilityDomain(percentile=1)
    ... )
    >>> pipe.fit(X_train)
    >>> predictions = pipe.predict(X_test)
    """

    def __init__(self, percentile=(0.1, 99.9)):
        if isinstance(percentile, (int, float)):
            if not 0 <= percentile <= 100:
                raise ValueError("percentile must be between 0 and 100")
            self.percentile = (percentile, 100 - percentile)
        else:
            if not all(0 <= p <= 100 for p in percentile):
                raise ValueError("percentiles must be between 0 and 100")
            if len(percentile) != 2:
                raise ValueError("percentile must be a float or tuple of 2 floats")
            if percentile[0] >= percentile[1]:
                raise ValueError("first percentile must be less than second")
            self.percentile = percentile

    def fit(self, X, y=None):
        """Fit the bounding box applicability domain.

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

        # Calculate bounds
        self.min_ = np.percentile(X, self.percentile[0], axis=0)
        self.max_ = np.percentile(X, self.percentile[1], axis=0)

        return self

    def transform(self, X):
        """Calculate the number of features outside their bounds for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        violations : ndarray of shape (n_samples, 1)
            Number of features outside their bounds for each sample.
            Zero indicates all features within bounds.
        """
        check_is_fitted(self)
        X = check_array(X)

        violations = np.sum((X < self.min_) | (X > self.max_), axis=1)
        return violations.reshape(-1, 1)

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
        violations = self.transform(X).ravel()
        return np.where(violations == 0, 1, -1)
