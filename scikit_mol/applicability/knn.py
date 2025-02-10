"""
K-Nearest Neighbors applicability domain.

This module was adapted from MLChemAD (https://github.com/OlivierBeq/MLChemAD)
Original work Copyright (c) 2023 Olivier J. M. Béquignon (MIT License)
Modifications Copyright (c) 2025 scikit-mol contributors (LGPL License)
See LICENSE.MIT in this directory for the original MIT license.
"""

from typing import Callable, ClassVar, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array

from .base import BaseApplicabilityDomain


class KNNApplicabilityDomain(BaseApplicabilityDomain):
    """Applicability domain defined using K-nearest neighbors.

    Determines domain membership based on the mean distance to k nearest neighbors
    in the training set. Higher distances indicate samples further from the
    training distribution.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use for distance calculation.
    percentile : float or None, default=None
        Percentile of training set distances to use as threshold (0-100).
        If None, uses 99.0 (include 99% of training samples).
    distance_metric : str or callable, default='euclidean'
        Distance metric to use. Options:
        - 'euclidean': Euclidean distance (default)
        - 'manhattan': Manhattan distance
        - 'cosine': Cosine distance
        - 'tanimoto': Tanimoto distance for binary fingerprints (same as 'jaccard')
        - 'jaccard': Jaccard distance for binary fingerprints
        - callable: Custom distance metric function(X, Y) -> array-like
        Any distance metric supported by sklearn.neighbors.NearestNeighbors can also be used.
        Note: Only distance metrics are supported (higher values = more distant) currently.
    n_jobs : int, default=None
        Number of parallel jobs to run for neighbors search.
        None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors.
    feature_name : str, default='KNN'
        Prefix for feature names in output.

    Notes
    -----
    For binary fingerprints, the Tanimoto distance is equivalent to the Jaccard distance.
    Both 'tanimoto' and 'jaccard' options use scipy's implementation of the Jaccard
    distance metric.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    threshold_ : float
        Distance threshold for domain membership.
    nn_ : NearestNeighbors
        Fitted nearest neighbors model.

    Examples
    --------
    >>> import numpy as np
    >>> from scikit_mol.applicability import KNNApplicabilityDomain
    >>>
    >>> # Generate example data
    >>> rng = np.random.RandomState(0)
    >>> X_train = rng.normal(0, 1, (100, 5))
    >>> X_test = rng.normal(0, 2, (20, 5))  # More spread out than training
    >>>
    >>> # Fit AD model
    >>> ad = KNNApplicabilityDomain(n_neighbors=5, percentile=95)
    >>> ad.fit(X_train)
    >>>
    >>> # Get raw distance scores (higher = more distant)
    >>> distances = ad.transform(X_test)
    >>>
    >>> # Get domain membership predictions
    >>> predictions = ad.predict(X_test)  # 1 = inside, -1 = outside
    >>>
    >>> # Get probability-like scores
    >>> scores = ad.score_transform(X_test)  # Higher = more likely inside
    """

    _scoring_convention: ClassVar[str] = (
        "high_outside"  # Higher distance = outside domain
    )

    def __init__(
        self,
        n_neighbors: int = 5,
        percentile: Optional[float] = None,
        distance_metric: Union[str, Callable] = "euclidean",
        n_jobs: Optional[int] = None,
        feature_name: str = "KNN",
    ) -> None:
        super().__init__(percentile=percentile, feature_name=feature_name)
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric
        self.n_jobs = n_jobs

    @property
    def distance_metric(self) -> Union[Callable, str]:
        return self._distance_metric

    @distance_metric.setter
    def distance_metric(self, value: Union[str, Callable]) -> None:
        if not isinstance(value, (str, Callable)):
            raise ValueError("distance_metric must be a string or callable")
        if value == "tanimoto":
            self._distance_metric = "jaccard"  # Use scipy's jaccard metric
        else:
            self._distance_metric = value

    def fit(self, X: ArrayLike, y=None) -> "KNNApplicabilityDomain":
        """Fit the KNN applicability domain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : KNNApplicabilityDomain
            Returns the instance itself.
        """
        if not isinstance(self.n_neighbors, int) or self.n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer")

        X = check_array(X, **self._check_params)
        self.n_features_in_ = X.shape[1]

        # Fit nearest neighbors model
        self.nn_ = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,  # +1 because point is its own neighbor
            metric=self.distance_metric,
            n_jobs=self.n_jobs,
        )
        self.nn_.fit(X)

        # Set initial threshold based on training data
        self.fit_threshold(X)

        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Calculate mean distance to k nearest neighbors in training set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Validated input data.

        Returns
        -------
        distances : ndarray of shape (n_samples, 1)
            Mean distance to k nearest neighbors. Higher values indicate samples
            further from the training set.
        """
        distances, _ = self.nn_.kneighbors(X)
        mean_distances = distances[:, 1:].mean(axis=1)  # Skip first (self) neighbor
        return mean_distances.reshape(-1, 1)
