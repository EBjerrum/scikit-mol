"""
Standardization approach applicability domain.

This module was adapted from MLChemAD (https://github.com/OlivierBeq/MLChemAD)
Original work Copyright (c) 2023 Olivier J. M. Béquignon (MIT License)
Modifications Copyright (c) 2025 scikit-mol contributors (LGPL License)
See LICENSE.MIT in this directory for the original MIT license.
"""

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array

from .base import BaseApplicabilityDomain


class StandardizationApplicabilityDomain(BaseApplicabilityDomain):
    """Applicability domain based on standardized feature values.

    Samples are considered within the domain if their standardized features
    fall within a certain number of standard deviations from the mean.
    The maximum absolute standardized value across all features is used
    as the score.

    Parameters
    ----------
    percentile : float or None, default=None
        Percentile of training set scores to use as threshold (0-100).
        If None, uses 95.0 (exclude top 5% of training samples).
    feature_name : str, default="Standardization"
        Name for the output feature column.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    scaler_ : StandardScaler
        Fitted standard scaler.
    threshold_ : float
        Current threshold for domain membership.

    Notes
    -----
    The scoring convention is 'high_outside' because higher standardized
    values indicate samples further from the training data mean.
    """

    _scoring_convention = "high_outside"

    def __init__(
        self,
        percentile: Optional[float] = None,
        feature_name: str = "Standardization",
    ) -> None:
        super().__init__(percentile=percentile or 95.0, feature_name=feature_name)

    def _set_statistical_threshold(self, X: NDArray) -> None:
        """Set threshold based on normal distribution.

        For normally distributed data, ~95% of values fall within
        2 standard deviations of the mean.
        """
        self.threshold_ = stats.norm.ppf(0.975)  # 2 standard deviations

    def fit(
        self, X: ArrayLike, y: Optional[Any] = None
    ) -> "StandardizationApplicabilityDomain":
        """Fit the standardization applicability domain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Any, optional (default=None)
            Not used, present for API consistency.

        Returns
        -------
        self : StandardizationApplicabilityDomain
            Returns the instance itself.
        """
        X = check_array(X, **self._check_params)
        self.n_features_in_ = X.shape[1]

        # Fit standard scaler
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X)

        # Set initial threshold based on training data
        self.fit_threshold(X)

        return self

    def _transform(self, X: NDArray) -> NDArray[np.float64]:
        """Calculate maximum absolute standardized values.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        scores : ndarray of shape (n_samples, 1)
            The maximum absolute standardized values. Higher values indicate
            samples further from the training data mean.
        """
        # Calculate standardized values and take max absolute value per sample
        X_std = self.scaler_.transform(X)
        scores = np.max(np.abs(X_std), axis=1)
        return scores.reshape(-1, 1)
