"""
Kernel Density applicability domain.

This module was adapted from [MLChemAD](https://github.com/OlivierBeq/MLChemAD)Chem
Original work Copyright (c) 2023 Olivier J. M. Béquignon (MIT License)
Modifications Copyright (c) 2025 scikit-mol contributors (LGPL License)
See LICENSE.MIT in this directory for the original MIT license.
"""

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.neighbors import KernelDensity
from sklearn.utils.validation import check_array

from .base import BaseApplicabilityDomain


class KernelDensityApplicabilityDomain(BaseApplicabilityDomain):
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
    percentile : float or None, default=None
        The percentile of training set densities to use as threshold (0-100).
        If None, uses 99.0 (exclude bottom 1% of training samples).
    feature_name : str, default="KernelDensity"
        Name for the output feature column.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    kde_ : KernelDensity
        Fitted kernel density estimator.
    threshold_ : float
        Density threshold for domain membership.

    Notes
    -----
    The scoring convention is 'high_inside' because higher density scores
    indicate samples more similar to the training data.

    Examples
    --------
    >>> from scikit_mol.applicability import KernelDensityApplicabilityDomain
    >>> ad = KernelDensityApplicabilityDomain(bandwidth=1.0)
    >>> ad.fit(X_train)
    >>> predictions = ad.predict(X_test)
    """

    _scoring_convention = "high_inside"

    def __init__(
        self,
        bandwidth: float = 1.0,
        kernel: str = "gaussian",
        percentile: Optional[float] = None,
        feature_name: str = "KernelDensity",
    ) -> None:
        super().__init__(percentile=percentile or 99.0, feature_name=feature_name)
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(
        self, X: ArrayLike, y: Optional[Any] = None
    ) -> "KernelDensityApplicabilityDomain":
        """Fit the kernel density applicability domain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Any, optional (default=None)
            Not used, present for API consistency.

        Returns
        -------
        self : KernelDensityApplicabilityDomain
            Returns the instance itself.
        """
        X = check_array(X, **self._check_params)
        self.n_features_in_ = X.shape[1]

        # Fit KDE
        self.kde_ = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
        self.kde_.fit(X)

        # Set initial threshold based on training data
        self.fit_threshold(X)

        return self

    def _transform(self, X: NDArray) -> NDArray[np.float64]:
        """Calculate log density scores for samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        scores : ndarray of shape (n_samples, 1)
            The log density scores of the samples. Higher scores indicate samples
            more similar to the training data.
        """
        scores = self.kde_.score_samples(X)
        return scores.reshape(-1, 1)
