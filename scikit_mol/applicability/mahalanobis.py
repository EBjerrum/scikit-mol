"""
Mahalanobis distance applicability domain.
"""

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, stats
from sklearn.utils.validation import check_array

from .base import BaseApplicabilityDomain


class MahalanobisApplicabilityDomain(BaseApplicabilityDomain):
    """Applicability domain based on Mahalanobis distance.

    Uses Mahalanobis distance to measure how many standard deviations a sample
    is from the training set mean, taking into account the covariance structure
    of the data. For multivariate normal data, the squared Mahalanobis distances
    follow a chi-square distribution.

    Parameters
    ----------
    percentile : float or None, default=None
        Percentile of training set scores to use as threshold (0-100).
        If None, uses 95.0 (exclude top 5% of training samples).
    feature_name : str, default="Mahalanobis"
        Name for the output feature column.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    mean_ : ndarray of shape (n_features,)
        Mean of training data.
    covariance_ : ndarray of shape (n_features, n_features)
        Covariance matrix of training data.
    threshold_ : float
        Current threshold for domain membership.

    Notes
    -----
    The scoring convention is 'high_outside' because higher Mahalanobis
    distances indicate samples further from the training data mean.
    """

    _scoring_convention = "high_outside"

    def __init__(
        self,
        percentile: Optional[float] = None,
        feature_name: str = "Mahalanobis",
    ) -> None:
        super().__init__(percentile=percentile or 95.0, feature_name=feature_name)

    def _set_statistical_threshold(self, X: NDArray) -> None:
        """Set threshold based on chi-square distribution.

        For multivariate normal data, squared Mahalanobis distances follow
        a chi-square distribution with degrees of freedom equal to the
        number of features.
        """
        df = self.n_features_in_
        self.threshold_ = np.sqrt(stats.chi2.ppf(0.95, df))

    def fit(
        self, X: ArrayLike, y: Optional[Any] = None
    ) -> "MahalanobisApplicabilityDomain":
        """Fit the Mahalanobis distance applicability domain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Any, optional (default=None)
            Not used, present for API consistency.

        Returns
        -------
        self : MahalanobisApplicabilityDomain
            Returns the instance itself.

        Raises
        ------
        ValueError
            If X has fewer samples than features, making covariance estimation unstable.
        """
        X = check_array(X, **self._check_params)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        if n_samples <= n_features:
            raise ValueError(
                f"n_samples ({n_samples}) must be greater than n_features ({n_features}) "
                "for stable covariance estimation."
            )

        # Calculate mean and covariance
        self.mean_ = np.mean(X, axis=0)
        self.covariance_ = np.cov(X, rowvar=False, ddof=1)

        # Add small regularization to ensure positive definiteness
        min_eig = np.min(linalg.eigvalsh(self.covariance_))
        if min_eig < 1e-6:
            self.covariance_ += (abs(min_eig) + 1e-6) * np.eye(n_features)

        # Set initial threshold based on training data
        self.fit_threshold(X)

        return self

    def _transform(self, X: NDArray) -> NDArray[np.float64]:
        """Calculate Mahalanobis distances.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        distances : ndarray of shape (n_samples, 1)
            The Mahalanobis distances of the samples. Higher distances indicate
            samples further from the training data mean.
        """
        # Calculate Mahalanobis distances using stable computation
        diff = X - self.mean_
        try:
            # Try Cholesky decomposition first (more stable)
            L = linalg.cholesky(self.covariance_, lower=True)
            mahal_dist = np.sqrt(
                np.sum(linalg.solve_triangular(L, diff.T, lower=True) ** 2, axis=0)
            )
        except linalg.LinAlgError:
            # Fallback to standard computation if Cholesky fails
            inv_covariance = linalg.pinv(
                self.covariance_
            )  # Use pseudo-inverse for stability
            mahal_dist = np.sqrt(np.sum(diff @ inv_covariance * diff, axis=1))

        return mahal_dist.reshape(-1, 1)
