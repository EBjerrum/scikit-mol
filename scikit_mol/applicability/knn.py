"""
K-Nearest Neighbors applicability domain.

This module was adapted from MLChemAD (https://github.com/OlivierBeq/MLChemAD)
Original work Copyright (c) 2023 Olivier J. M. Béquignon (MIT License)
Modifications Copyright (c) 2025 scikit-mol contributors (LGPL License)
See LICENSE.MIT in this directory for the original MIT license.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array, check_is_fitted


class KNNApplicabilityDomain(BaseEstimator, TransformerMixin):
    """Applicability domain defined using K-nearest neighbors.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use for distance calculation.
    percentile : float, default=99
        Percentile of training set distances to use as threshold.
        Samples with distances above this percentile are considered outside
        the applicability domain. The fit_threshold method can be used to update
        the threshold using new data without refitting the model (e.g. validation data).
    metric : str, default='euclidean'
        Distance metric to use for nearest neighbor calculation.
        Any metric supported by sklearn.neighbors.NearestNeighbors can be used.
    n_jobs : int, default=None
        Number of parallel jobs to run for neighbors search.
        None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    threshold_ : float
        Distance threshold for the applicability domain.
    nn_ : NearestNeighbors
        Fitted nearest neighbors model.
    """

    def __init__(self, n_neighbors=5, percentile=95, metric="euclidean", n_jobs=None):
        self.n_neighbors = n_neighbors
        self.percentile = percentile
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit the KNN applicability domain.

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
        if not 0 <= self.percentile <= 100:
            raise ValueError("percentile must be between 0 and 100")

        X = check_array(X, accept_sparse=True)

        self.n_features_in_ = X.shape[1]

        # Fit nearest neighbors model
        self.nn_ = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,  # +1 because point is its own neighbor
            metric=self.metric,
            n_jobs=self.n_jobs,
        )
        self.nn_.fit(X)

        # Set initial threshold based on training data
        self.fit_threshold(X)

        return self

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
        X = check_array(X, accept_sparse=True)

        # Calculate distances to k nearest neighbors
        distances, _ = self.nn_.kneighbors(X)
        mean_distances = distances[:, 1:].mean(axis=1)

        # Set threshold based on distance distribution
        self.threshold_ = np.percentile(mean_distances, self.percentile)

        return self

    def transform(self, X):
        """Calculate mean distance to k nearest neighbors in training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        distances : ndarray of shape (n_samples, 1)
            Mean distance to k nearest neighbors. Higher values indicate samples
            further from the training set.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=True)

        # Calculate distances to k nearest neighbors
        distances, _ = self.nn_.kneighbors(X)
        mean_distances = distances.mean(axis=1)

        return mean_distances.reshape(-1, 1)

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
