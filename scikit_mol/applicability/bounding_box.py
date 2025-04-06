"""
Bounding box applicability domain.

This module was adapted from [MLChemAD](https://github.com/OlivierBeq/MLChemAD)
Original work Copyright (c) 2023 Olivier J. M. Béquignon (MIT License)
Modifications Copyright (c) 2025 scikit-mol contributors (LGPL License)
See LICENSE.MIT in this directory for the original MIT license.
"""

from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import BaseApplicabilityDomain


class BoundingBoxApplicabilityDomain(BaseApplicabilityDomain):
    """Applicability domain defined by feature value ranges.

    Samples falling outside the allowed range for any feature are considered
    outside the domain. The range for each feature is defined by percentiles
    of the training set distribution.

    Parameters
    ----------
    percentile : float or tuple of float, default=(0.1, 99.9)
        Percentile(s) of the training set distribution used to define
        the bounding box. If float, uses (percentile, 100-percentile).
    feature_name : str, default="BoundingBox"
        Prefix for feature names in output.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    min_ : ndarray of shape (n_features,)
        Minimum allowed value for each feature.
    max_ : ndarray of shape (n_features,)
        Maximum allowed value for each feature.
    threshold_ : float
        Current threshold for domain membership (always 0.5).

    Notes
    -----
    The bounding box method is simple but effective, especially for chemical
    descriptors with clear physical interpretations. For high-dimensional or
    correlated features, other methods may be more appropriate.

    Examples
    --------
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from scikit_mol.applicability import BoundingBoxApplicabilityDomain
    >>>
    >>> # Basic usage
    >>> ad = BoundingBoxApplicabilityDomain(percentile=1)
    >>> ad.fit(X_train)
    >>> predictions = ad.predict(X_test)
    >>>
    >>> # With preprocessing
    >>> pipe = make_pipeline(
    ...     StandardScaler(),
    ...     BoundingBoxApplicabilityDomain(percentile=1)
    ... )
    >>> pipe.fit(X_train)
    >>> predictions = pipe.predict(X_test)
    """

    _scoring_convention = "high_outside"
    _supports_threshold_fitting = False

    def __init__(
        self,
        percentile: Union[float, Tuple[float, float]] = (0.1, 99.9),
        feature_name: str = "BoundingBox",
    ) -> None:
        super().__init__(percentile=None, feature_name=feature_name)

        if isinstance(percentile, (int, float)):
            if not 0 <= percentile <= 100:
                raise ValueError("percentile must be between 0 and 100")
            self.box_percentile = (percentile, 100 - percentile)
        else:
            if not all(0 <= p <= 100 for p in percentile):
                raise ValueError("percentiles must be between 0 and 100")
            if len(percentile) != 2:
                raise ValueError("percentile must be a float or tuple of 2 floats")
            if percentile[0] >= percentile[1]:
                raise ValueError("first percentile must be less than second")
            self.box_percentile = percentile

    def fit(
        self, X: ArrayLike, y: Optional[Any] = None
    ) -> "BoundingBoxApplicabilityDomain":
        """Fit the bounding box applicability domain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : BoundingBoxApplicabilityDomain
            Returns the instance itself.
        """
        X = self._validate_data(X)
        self.n_features_in_ = X.shape[1]

        # Calculate bounds
        self.min_ = np.percentile(X, self.box_percentile[0], axis=0)
        self.max_ = np.percentile(X, self.box_percentile[1], axis=0)

        # Fixed threshold since we count violations
        self.threshold_ = 0.5

        return self

    def _transform(self, X: NDArray) -> NDArray[np.float64]:
        """Calculate the number of features outside their bounds.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        violations : ndarray of shape (n_samples, 1)
            Number of features outside their bounds for each sample.
            Zero indicates all features within bounds.
        """
        violations = np.sum((X < self.min_) | (X > self.max_), axis=1)
        return violations.reshape(-1, 1)
