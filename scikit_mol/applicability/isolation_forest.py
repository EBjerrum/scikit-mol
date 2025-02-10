"""
Isolation Forest applicability domain.

This module was adapted from MLChemAD (https://github.com/OlivierBeq/MLChemAD)
Original work Copyright (c) 2023 Olivier J. M. Béquignon (MIT License)
Modifications Copyright (c) 2025 scikit-mol contributors (LGPL License)
See LICENSE.MIT in this directory for the original MIT license.
"""

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import IsolationForest
from sklearn.utils.validation import check_array

from .base import BaseApplicabilityDomain


class IsolationForestApplicabilityDomain(BaseApplicabilityDomain):
    """Applicability domain based on Isolation Forest.

    Uses Isolation Forest to identify outliers based on the isolation depth
    of samples in random decision trees.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    contamination : float, default=0.01
        Expected proportion of outliers in the training data.
    random_state : Optional[int], default=None
        Controls the randomness of the forest.
    percentile : float or None, default=None
        Percentile of training set scores to use as threshold (0-100).
        If None, uses contamination-based threshold from IsolationForest.
    feature_name : str, default="IsolationForest"
        Name for feature names in output.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    iforest_ : IsolationForest
        Fitted isolation forest model.
    threshold_ : float
        Current threshold for domain membership.

    Notes
    -----
    The scoring convention is 'high_inside' because higher scores from
    IsolationForest indicate samples more similar to the training data.

    References
    ----------
    .. [1] Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest.
           In 2008 Eighth IEEE International Conference on Data Mining (pp. 413-422).
    """

    _scoring_convention = "high_inside"
    _supports_threshold_fitting = True

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.01,
        random_state: Optional[int] = None,
        percentile: Optional[float] = None,
        feature_name: str = "IsolationForest",
    ) -> None:
        if not 0 < contamination < 1:
            raise ValueError("contamination must be between 0 and 1")
        super().__init__(percentile=percentile, feature_name=feature_name)
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state

    def fit(
        self, X: ArrayLike, y: Optional[Any] = None
    ) -> "IsolationForestApplicabilityDomain":
        """Fit the isolation forest applicability domain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : IsolationForestApplicabilityDomain
            Returns the instance itself.
        """
        X = check_array(X, **self._check_params)
        self.n_features_in_ = X.shape[1]

        self.iforest_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.iforest_.fit(X)

        # Set initial threshold
        if self.percentile is not None:
            self.fit_threshold(X)
        else:
            # Use IsolationForest's default threshold
            self.threshold_ = self.iforest_.offset_

        return self

    def _transform(self, X: NDArray) -> NDArray[np.float64]:
        """Calculate anomaly scores for samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        scores : ndarray of shape (n_samples, 1)
            The anomaly scores of the samples.
            Higher scores indicate samples more similar to training data.
        """
        scores = self.iforest_.score_samples(X)
        return scores.reshape(-1, 1)

    # def fit_threshold(self, X, target_percentile=95):
    #     """Update the threshold using new data without refitting the model.

    #     Parameters
    #     ----------
    #     X : array-like of shape (n_samples, n_features)
    #         Data to compute threshold from.
    #     target_percentile : float, default=95
    #         Target percentile of samples to include within domain.

    #     Returns
    #     -------
    #     self : object
    #         Returns the instance itself.
    #     """
    #     check_is_fitted(self)
    #     X = check_array(X)

    #     if not 0 <= target_percentile <= 100:
    #         raise ValueError("target_percentile must be between 0 and 100")

    #     # Get decision function scores
    #     scores = self.iforest_.score_samples(X)

    #     # Set threshold to achieve desired percentile
    #     self.threshold_ = np.percentile(scores, 100 - target_percentile)

    #     return self

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
    #     if hasattr(self, "threshold_"):
    #         return np.where(scores > self.threshold_, 1, -1)
    #     return self.iforest_.predict(X)
