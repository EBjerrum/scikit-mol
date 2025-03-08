"""
TOPKAT's Optimal Prediction Space (OPS) applicability domain.

This module was adapted from MLChemAD (https://github.com/OlivierBeq/MLChemAD)
Original work Copyright (c) 2023 Olivier J. M. Béquignon (MIT License)
Modifications Copyright (c) 2025 scikit-mol contributors (LGPL License)
See LICENSE.MIT in this directory for the original MIT license.

"""

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.validation import check_array

from .base import BaseApplicabilityDomain


class TopkatApplicabilityDomain(BaseApplicabilityDomain):
    """Applicability domain defined using TOPKAT's Optimal Prediction Space (OPS).

    The method transforms the input space (P-space) to a normalized space (S-space),
    then projects it to the Optimal Prediction Space using eigendecomposition.

    Parameters
    ----------
    percentile : float or None, default=None
        Not used, present for API consistency.
    feature_name : str, default="TOPKAT"
        Name for the output feature column.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    X_min_ : ndarray of shape (n_features,)
        Minimum values of training features.
    X_max_ : ndarray of shape (n_features,)
        Maximum values of training features.
    eigen_val_ : ndarray of shape (n_features + 1,)
        Eigenvalues of the S-space transformation.
    eigen_vec_ : ndarray of shape (n_features + 1, n_features + 1)
        Eigenvectors of the S-space transformation.
    threshold_ : float
        Fixed threshold based on dimensionality.

    Notes
    -----
    The scoring convention is 'high_outside' because higher OPS distances
    indicate samples further from the training data.

    References
    ----------
    .. [1] Gombar, Vijay K. (1996). Method and apparatus for validation of model-based
           predictions (US Patent No. 6-036-349) USPTO.
    """

    _scoring_convention = "high_outside"
    _supports_threshold_fitting = False

    def __init__(
        self,
        percentile: Optional[float] = None,
        feature_name: str = "TOPKAT",
    ) -> None:
        super().__init__(percentile=None, feature_name=feature_name)

    def fit(self, X: ArrayLike, y: Optional[Any] = None) -> "TopkatApplicabilityDomain":
        """Fit the TOPKAT applicability domain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Any, optional (default=None)
            Not used, present for API consistency.

        Returns
        -------
        self : TopkatApplicabilityDomain
            Returns the instance itself.
        """
        X = check_array(X, **self._check_params)
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]

        # Store scaling factors
        self.X_min_ = X.min(axis=0)
        self.X_max_ = X.max(axis=0)

        # Transform P-space to S-space
        denom = np.where(
            (self.X_max_ - self.X_min_) != 0, (self.X_max_ - self.X_min_), 1
        )
        S = (2 * X - self.X_max_ - self.X_min_) / denom

        # Add column of ones
        S = np.c_[np.ones(n_samples), S]

        # Calculate eigendecomposition
        self.eigen_val_, self.eigen_vec_ = np.linalg.eigh(S.T.dot(S))

        # Ensure real values (numerical stability)
        self.eigen_val_ = np.real(self.eigen_val_)
        self.eigen_vec_ = np.real(self.eigen_vec_)

        # Set fixed threshold based on dimensionality
        self.threshold_ = 5 * (self.n_features_in_ + 1) / (2 * self.n_features_in_)

        return self

    def _transform(self, X: NDArray) -> NDArray[np.float64]:
        """Calculate OPS distance scores for samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        distances : ndarray of shape (n_samples, 1)
            OPS distance scores. Higher values indicate samples further
            from the training data.
        """
        # Transform to S-space
        denom = np.where(
            (self.X_max_ - self.X_min_) != 0, (self.X_max_ - self.X_min_), 1
        )
        S = (2 * X - self.X_max_ - self.X_min_) / denom

        # Add column of ones
        if X.ndim == 1:
            S = np.r_[1, S].reshape(1, -1)
        else:
            S = np.c_[np.ones(X.shape[0]), S]

        # Project to OPS
        OPS = S.dot(self.eigen_vec_)

        # Calculate OPS distances - matching MLChemAD's approach
        denom = np.divide(
            np.ones_like(self.eigen_val_, dtype=float),
            self.eigen_val_,
            out=np.zeros_like(self.eigen_val_),
            where=self.eigen_val_ != 0,
        )
        distances = (OPS * OPS).dot(denom)

        return distances.reshape(-1, 1)

    # def predict(self, X):
    #     """Predict whether samples are within the applicability domain.

    #     Parameters
    #     ----------
    #     X : array-like of shape (n_samples, n_features)
    #         The samples to predict.

    #     Returns
    #     -------
    #     y_pred : ndarray of shape (n_samples,)
    #         Returns 1 for samples inside the domain and -1 for samples outside
    #         (following scikit-learn's convention for outlier detection).
    #     """
    #     scores = self._transform(X).ravel()
    #     threshold = self.threshold_
    #     return np.where(scores < threshold, 1, -1)
