"""
Hotelling T² applicability domain.

This module was adapted from [MLChemAD](https://github.com/OlivierBeq/MLChemAD)
Original work Copyright (c) 2023 Olivier J. M. Béquignon (MIT License)
Modifications Copyright (c) 2025 scikit-mol contributors (LGPL License)
See LICENSE.MIT in this directory for the original MIT license.
"""

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import f as f_dist
from sklearn.utils import check_array

from .base import BaseApplicabilityDomain


class HotellingT2ApplicabilityDomain(BaseApplicabilityDomain):
    """Applicability domain based on Hotelling's T² statistic.

    Uses Hotelling's T² statistic to define an elliptical confidence region
    around the training data. The threshold can be set using either the
    F-distribution (statistical approach) or adjusted using a validation set.

    Parameters
    ----------
    significance : float, default=0.05
        Significance level for F-distribution threshold.
    percentile : float or None, default=None
        If not None, overrides significance-based threshold.
        Must be between 0 and 100.
    feature_name : str, default="HotellingT2"
        Prefix for feature names in output.

    Notes
    -----
    Lower volume protrusion scores indicate samples closer to the training
    data center. By default, the threshold is set using the F-distribution
    with a significance level of 0.05 (95% confidence).

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    t2_ : ndarray of shape (n_features,)
        Hotelling T² ellipse parameters.
    threshold_ : float
        Current threshold for volume protrusions.

    References
    ----------
    .. [1] Hotelling, H. (1931). The generalization of Student's ratio.
           The Annals of Mathematical Statistics, 2(3), 360-378.
    """

    _scoring_convention = "high_outside"
    _supports_threshold_fitting = True

    def __init__(
        self,
        significance: float = 0.05,
        percentile: Optional[float] = None,
        feature_name: str = "HotellingT2",
    ) -> None:
        if not 0 < significance < 1:
            raise ValueError("significance must be between 0 and 1")
        super().__init__(percentile=percentile, feature_name=feature_name)
        self.significance = significance

    def _set_statistical_threshold(self, X: NDArray) -> None:
        """Set threshold using F-distribution."""
        n_samples = X.shape[0]
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

    def fit(
        self, X: ArrayLike, y: Optional[Any] = None
    ) -> "HotellingT2ApplicabilityDomain":
        """Fit the Hotelling T² applicability domain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : HotellingT2ApplicabilityDomain
            Returns the instance itself.
        """
        X = check_array(X, **self._check_params)
        self.n_features_in_ = X.shape[1]

        # Determine the Hotelling T² ellipse
        self.t2_ = np.sqrt((1 / X.shape[0]) * (X**2).sum(axis=0))

        # Set initial threshold
        if self.percentile is not None:
            self.fit_threshold(X)
        else:
            self._set_statistical_threshold(X)

        return self

    def _transform(self, X: NDArray) -> NDArray[np.float64]:
        """Calculate volume protrusion scores for samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        scores : ndarray of shape (n_samples, 1)
            The volume protrusion scores. Higher values indicate samples
            further from the training data center.
        """
        protrusions = (X**2 / self.t2_**2).sum(axis=1)
        return protrusions.reshape(-1, 1)
