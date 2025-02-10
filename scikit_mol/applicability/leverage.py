"""
Leverage-based applicability domain.

This module was adapted from MLChemAD (https://github.com/OlivierBeq/MLChemAD) as described in the README.md file.
Original work Copyright (c) 2023 Olivier J. M. Béquignon (MIT License)
Modifications Copyright (c) 2025 scikit-mol contributors (LGPL License)
See LICENSE.MIT in this directory for the original MIT license.
"""

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.validation import check_array

from .base import BaseApplicabilityDomain


class LeverageApplicabilityDomain(BaseApplicabilityDomain):
    """Applicability domain defined using the leverage approach.

    The leverage approach measures how far a sample is from the center of the
    feature space using the diagonal elements of the hat matrix H = X(X'X)^(-1)X'.
    Higher leverage values indicate samples further from the center of the training data.

    Parameters
    ----------
    threshold_factor : float, default=3
        Factor used in calculating the leverage threshold h* = threshold_factor * (p+1)/n
        where p is the number of features and n is the number of samples.
    percentile : float or None, default=None
        If not None, overrides the statistical threshold with a percentile-based one.
        See BaseApplicabilityDomain for details.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    threshold_ : float
        Calculated leverage threshold.
    var_covar_ : ndarray of shape (n_features, n_features)
        Variance-covariance matrix of the training data.

    Notes
    -----
    The statistical threshold h* = 3 * (p+1)/n is a commonly used rule of thumb
    in regression diagnostics, where p is the number of features and n is the
    number of training samples.

    Input data should be scaled (e.g., using StandardScaler) to ensure all features
    contribute equally. For high-dimensional data like fingerprints, dimensionality
    reduction (e.g., PCA) is strongly recommended to avoid computational issues with
    the variance-covariance matrix inversion.

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.decomposition import PCA
    >>> from scikit_mol.applicability import LeverageApplicabilityDomain
    >>>
    >>> # Create pipeline with scaling and dimensionality reduction
    >>> pipe = Pipeline([
    ...     ('scaler', StandardScaler()),
    ...     ('pca', PCA(n_components=0.95)),  # Keep 95% of variance
    ...     ('ad', LeverageApplicabilityDomain())
    ... ])
    >>>
    >>> # Fit pipeline
    >>> X_train = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]  # Example data
    >>> pipe.fit(X_train)
    >>>
    >>> # Predict domain membership for new samples
    >>> X_test = [[0, 1, 2], [10, 20, 30]]
    >>> pipe.predict(X_test)  # Returns [1, -1] (in/out of domain)
    """

    _scoring_convention = "high_outside"
    _supports_threshold_fitting = True

    def __init__(
        self,
        threshold_factor: float = 3,
        percentile: Optional[float] = None,
        feature_name: str = "Leverage",
    ) -> None:
        super().__init__(percentile=percentile, feature_name=feature_name)
        self.threshold_factor = threshold_factor

    def _set_statistical_threshold(self, X: NDArray) -> None:
        """Set the statistical threshold h* = threshold_factor * (p+1)/n."""
        n_samples = X.shape[0]
        self.threshold_ = self.threshold_factor * (self.n_features_in_ + 1) / n_samples

    def fit(
        self, X: ArrayLike, y: Optional[Any] = None
    ) -> "LeverageApplicabilityDomain":
        """Fit the leverage applicability domain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : LeverageApplicabilityDomain
            Returns the instance itself.
        """
        X = check_array(X, **self._check_params)
        self.n_features_in_ = X.shape[1]

        # Calculate variance-covariance matrix
        self.var_covar_ = np.linalg.inv(X.T.dot(X))

        # Set initial threshold
        self._set_statistical_threshold(X)

        return self

    def _transform(self, X: NDArray) -> NDArray[np.float64]:
        """Calculate leverage values.

        Higher values indicate samples further from the center of the training data.
        """
        h = np.sum(X.dot(self.var_covar_) * X, axis=1)
        return h.reshape(-1, 1)
