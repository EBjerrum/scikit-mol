"""Base class for applicability domain estimators."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils._set_output import _SetOutputMixin, _wrap_method_output
from sklearn.utils.validation import check_is_fitted


class _ADOutputMixin(_SetOutputMixin):
    """Extends sklearn's _SetOutputMixin to handle predict and score_transform methods."""

    def __init_subclass__(cls, **kwargs):
        # First handle transform/fit_transform via parent
        super().__init_subclass__(auto_wrap_output_keys=("transform",), **kwargs)

        # Add our additional methods
        for method in ["predict", "score_transform"]:
            if method not in cls.__dict__:
                continue
            wrapped_method = _wrap_method_output(getattr(cls, method), "transform")
            setattr(cls, method, wrapped_method)


class BaseApplicabilityDomain(BaseEstimator, TransformerMixin, _ADOutputMixin, ABC):
    """Base class for applicability domain estimators.

    Parameters
    ----------
    percentile : float or None, default=None
        Percentile of samples to consider within domain (0-100).
        If None:
        - For methods with statistical thresholds: use statistical method
        - For percentile-only methods: use 99.0 (include 99% of training samples)

    Notes
    -----
    Subclasses must define _scoring_convention as either:
    - 'high_outside': Higher scores indicate samples outside domain (e.g., distances)
    - 'high_inside': Higher scores indicate samples inside domain (e.g., likelihoods)

    The raw scores from transform() should maintain their natural interpretation,
    while predict() will handle the conversion to ensure consistent output
    (1 = inside domain, -1 = outside domain).

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    threshold_ : float
        Current threshold for domain membership.
    """

    _supports_threshold_fitting: ClassVar[bool] = True
    _scoring_convention: ClassVar[str]  # Must be set by subclasses

    def __init__(
        self, percentile: Optional[float] = None, feature_prefix: str = "AD_estimator"
    ) -> None:
        if not hasattr(self, "_scoring_convention"):
            raise TypeError(
                f"Class {self.__class__.__name__} must define _scoring_convention "
                "as either 'high_outside' or 'high_inside'"
            )
        if self._scoring_convention not in ["high_outside", "high_inside"]:
            raise ValueError(
                f"Invalid _scoring_convention '{self._scoring_convention}'. "
                "Must be either 'high_outside' or 'high_inside'"
            )
        if percentile is not None and not 0 <= percentile <= 100:
            raise ValueError("percentile must be between 0 and 100")
        self.percentile = percentile
        self.feature_prefix = feature_prefix
        self._check_params = {
            "estimator": self,
            "accept_sparse": False,
            "dtype": None,
            "force_all_finite": True,
            "ensure_2d": True,
        }

    @abstractmethod
    def fit(self, X: ArrayLike, y: Optional[Any] = None) -> "BaseApplicabilityDomain":
        """Fit the applicability domain estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Any, optional (default=None)
            Not used, present for API consistency.

        Returns
        -------
        self : BaseApplicabilityDomain
            Returns the instance itself.
        """
        raise NotImplementedError("Subclasses should implement fit")

    def fit_threshold(
        self, X: ArrayLike, target_percentile: Optional[float] = None
    ) -> "BaseApplicabilityDomain":
        """Update threshold estimation using new data.

        Parameters
        ----------
        X : array-like
            Data to compute threshold from.
        target_percentile : float, optional (default=None)
            If provided: Use this percentile and update self.percentile
            If None: Use current self.percentile setting
            - For methods with statistical thresholds: use statistical method if percentile=None
            - For percentile-only methods: use 99.0 if percentile=None

        Returns
        -------
        self : BaseApplicabilityDomain
            Returns the instance itself.
        """
        check_is_fitted(self)
        X = check_array(X, **self._check_params)

        if target_percentile is not None:
            if not 0 <= target_percentile <= 100:
                raise ValueError("target_percentile must be between 0 and 100")
            self.percentile = target_percentile

        # Use statistical threshold if available and percentile is None
        if self.percentile is None:
            if hasattr(self, "_set_statistical_threshold"):
                self._set_statistical_threshold(X)
            else:
                # Use 99th percentile for methods without statistical thresholds
                scores = self.transform(X).ravel()
                if self._scoring_convention == "high_outside":
                    self.threshold_ = np.percentile(
                        scores, 99.0
                    )  # Only 1% above threshold (outside)
                else:  # high_inside
                    self.threshold_ = np.percentile(
                        scores, 1.0
                    )  # Only 1% below threshold (outside)
        else:
            scores = self.transform(X).ravel()
            if self._scoring_convention == "high_outside":
                self.threshold_ = np.percentile(
                    scores, self.percentile
                )  # percentile% below = inside
            else:  # high_inside
                self.threshold_ = np.percentile(
                    scores, 100 - self.percentile
                )  # percentile% above = inside

        return self

    def transform(
        self, X: Union[ArrayLike, pd.DataFrame], y: Optional[Any] = None
    ) -> Union[NDArray[np.float64], pd.DataFrame]:
        """Calculate applicability domain scores.

        Parameters
        ----------
        X : array-like or pandas DataFrame
            The data to transform.

        Returns
        -------
        scores : ndarray or pandas DataFrame
            Method-specific scores. Interpretation depends on _scoring_convention:
            - 'high_outside': Higher scores indicate samples further from training data
            - 'high_inside': Higher scores indicate samples closer to training data
            Shape (n_samples, 1).
        """
        check_is_fitted(self)
        X = check_array(X, **self._check_params)

        # Calculate scores
        scores = self._transform(X)

        return scores

    @abstractmethod
    def _transform(self, X: NDArray) -> NDArray[np.float64]:
        """Implementation of the transform method.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Validated input data.

        Returns
        -------
        scores : ndarray of shape (n_samples, 1)
            Method-specific scores.
        """
        raise NotImplementedError("Subclasses should implement _transform")

    def predict(
        self, X: Union[ArrayLike, pd.DataFrame]
    ) -> Union[NDArray[np.int_], pd.DataFrame]:
        """Predict whether samples are within the applicability domain."""

        check_is_fitted(self)
        X = check_array(X, **self._check_params)

        # Calculate predictions
        scores = self._transform(X).ravel()
        if self._scoring_convention == "high_outside":
            predictions = np.where(scores <= self.threshold_, 1, -1)
        else:  # high_inside
            predictions = np.where(scores >= self.threshold_, 1, -1)

        return predictions

    def score_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Transform raw scores to [0,1] range using sigmoid.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The samples to transform.

        Returns
        -------
        scores : ndarray of shape (n_samples, 1)
            Transformed scores in [0,1] range. Higher values indicate
            samples more likely to be within domain, regardless of
            the method's raw score convention.
        """

        check_is_fitted(self)
        scores = self.transform(
            X
        )  # May be pandas dataframe returned if that is set as output transform.
        scores = check_array(scores, **self._check_params).ravel()

        # TODO: the sharpness ought to somehow be fitted to the range of the raw_scores
        if self._scoring_convention == "high_outside":
            # Flip sign for sigmoid so higher output = more likely inside
            return (1 / (1 + np.exp(scores - self.threshold_))).reshape(-1, 1)
        else:  # high_inside
            # No sign flip needed
            return (1 / (1 + np.exp(self.threshold_ - scores))).reshape(-1, 1)

    def get_feature_names_out(self) -> NDArray[np.str_]:
        """Get feature name for output column."""

        return np.array([f"{self.feature_prefix}"])
