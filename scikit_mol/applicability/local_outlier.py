"""
Local Outlier Factor applicability domain.

This module was adapted from MLChemAD (https://github.com/OlivierBeq/MLChemAD)
Original work Copyright (c) 2023 Olivier J. M. Béquignon (MIT License)
Modifications Copyright (c) 2025 scikit-mol contributors (LGPL License)
See LICENSE.MIT in this directory for the original MIT license.
"""

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils.validation import check_array

from .base import BaseApplicabilityDomain


class LocalOutlierFactorApplicabilityDomain(BaseApplicabilityDomain):
    """Applicability domain based on Local Outlier Factor (LOF).

    LOF measures the local deviation of density of a sample with respect to its
    neighbors, identifying samples that have substantially lower density than
    their neighbors.

    Parameters
    ----------
    n_neighbors : int, default=20
        Number of neighbors to use for LOF calculation.
    contamination : float, default=0.1
        Expected proportion of outliers in the data set.
    metric : str, default='euclidean'
        Metric to use for distance computation.
    percentile : float or None, default=None
        Percentile of training set scores to use as threshold (0-100).
        If None, uses contamination-based threshold from LOF.
    feature_name : str, default="LOF"
        Name for the output feature column.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    lof_ : LocalOutlierFactor
        Fitted LOF estimator.
    threshold_ : float
        Current threshold for domain membership.

    Notes
    -----
    The scoring convention is 'high_outside' because higher LOF scores
    indicate samples that are more likely to be outliers.

    References
    ----------
    .. [1] Breunig et al. (2000). LOF: Identifying Density-Based Local Outliers.
           In: Proc. 2000 ACM SIGMOD Int. Conf. Manag. Data, ACM, pp. 93-104.
    """

    _scoring_convention = "high_outside"

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.1,
        metric: str = "euclidean",
        percentile: Optional[float] = None,
        feature_name: str = "LOF",
    ) -> None:
        super().__init__(percentile=percentile, feature_name=feature_name)
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric

    def fit(
        self, X: ArrayLike, y: Optional[Any] = None
    ) -> "LocalOutlierFactorApplicabilityDomain":
        """Fit the LOF applicability domain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Any, optional (default=None)
            Not used, present for API consistency.

        Returns
        -------
        self : LocalOutlierFactorApplicabilityDomain
            Returns the instance itself.
        """
        X = check_array(X, **self._check_params)
        self.n_features_in_ = X.shape[1]

        self.lof_ = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            contamination=self.contamination,
            novelty=True,
        )
        self.lof_.fit(X)

        # Set initial threshold based on training data
        self.fit_threshold(X)

        return self

    def _transform(self, X: NDArray) -> NDArray[np.float64]:
        """Calculate LOF scores for samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        scores : ndarray of shape (n_samples, 1)
            The LOF scores of the samples. Higher scores indicate samples
            that are more likely to be outliers.
        """
        # Get negative LOF scores (higher means more likely to be outlier)
        scores = -self.lof_.score_samples(X)
        return scores.reshape(-1, 1)

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
    #     return self.lof_.predict(X)
