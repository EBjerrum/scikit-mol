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


def _safe_flatten(X: Union[ArrayLike, pd.DataFrame]) -> NDArray[np.float64]:
    """Safely flatten numpy arrays or pandas DataFrames to 1D array.

    Parameters
    ----------
    X : array-like or DataFrame of shape (n_samples, n_features)
        Input data to flatten

    Returns
    -------
    flattened : ndarray of shape (n_samples,)
        Flattened 1D array
    """
    if hasattr(X, "to_numpy"):  # pandas DataFrame
        return X.to_numpy().ravel()
    return np.asarray(X).ravel()


class BaseApplicabilityDomain(BaseEstimator, TransformerMixin, _ADOutputMixin, ABC):
    """Base class for applicability domain estimators.

    Parameters
    ----------
    percentile : float or None, default=None
        Percentile of samples to consider within domain (0-100).
        If None:
        - For methods with statistical thresholds: use statistical method
        - For percentile-only methods: use 99.0 (include 99% of training samples)
    feature_name : str, default="AD_estimator"
        Name for the output feature column.

    Notes
    -----
    Subclasses must define `_scoring_convention` as either:
    - 'high_outside': Higher scores indicate samples outside domain (e.g., distances)
    - 'high_inside': Higher scores indicate samples inside domain (e.g., likelihoods)

    The raw scores from `.transform()` should maintain their natural interpretation,
    while `.predict()` will handle the conversion to ensure consistent output
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
        self, percentile: Optional[float] = None, feature_name: str = "AD_estimator"
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
        self.feature_name = feature_name
        self._check_params = {
            "estimator": self,
            "accept_sparse": False,
            "dtype": None,
            "ensure_all_finite": True,
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
        self,
        X: Union[ArrayLike, pd.DataFrame],
        target_percentile: Optional[float] = None,
    ) -> "BaseApplicabilityDomain":
        """Update threshold estimation using new data."""
        check_is_fitted(self)
        X = check_array(X, **self._check_params)

        if target_percentile is not None:
            if not 0 <= target_percentile <= 100:
                raise ValueError("target_percentile must be between 0 and 100")
            self.percentile = target_percentile

        # Use statistical threshold if available and percentile is None
        if self.percentile is None and hasattr(self, "_set_statistical_threshold"):
            self._set_statistical_threshold(X)
            return self

        # Otherwise use percentile-based threshold
        scores = _safe_flatten(self.transform(X))

        if self.percentile is None:
            # Default percentile for methods without statistical thresholds
            if self._scoring_convention == "high_outside":
                self.threshold_ = np.percentile(scores, 99.0)
            else:  # high_inside
                self.threshold_ = np.percentile(scores, 1.0)
        else:
            if self._scoring_convention == "high_outside":
                self.threshold_ = np.percentile(scores, self.percentile)
            else:  # high_inside
                self.threshold_ = np.percentile(scores, 100 - self.percentile)

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
            Method-specific scores. Interpretation depends on `_scoring_convention`:
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
        """Predict whether samples are within the applicability domain.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Returns 1 for inside and -1 for outside.
        """
        check_is_fitted(self)
        X = check_array(X, **self._check_params)

        # Calculate predictions
        scores = _safe_flatten(self.transform(X))
        if self._scoring_convention == "high_outside":
            predictions = np.where(scores <= self.threshold_, 1, -1)
        else:  # high_inside
            predictions = np.where(scores >= self.threshold_, 1, -1)

        return predictions.ravel()

    def score_transform(
        self, X: Union[ArrayLike, pd.DataFrame]
    ) -> Union[NDArray[np.float64], pd.DataFrame]:
        """Transform raw scores to [0,1] range using sigmoid.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            The samples to transform.

        Returns
        -------
        scores : ndarray or DataFrame of shape (n_samples, 1)
            Transformed scores in [0,1] range. Higher values indicate
            samples more likely to be within domain, regardless of
            the method's raw score convention.
        """
        check_is_fitted(self)
        scores = _safe_flatten(self.transform(X))

        # TODO: the sharpness ought to somehow be fitted to the range of the raw_scores
        if self._scoring_convention == "high_outside":
            # Flip sign for sigmoid so higher output = more likely inside
            return (1 / (1 + np.exp(scores - self.threshold_))).reshape(-1, 1)
        else:  # high_inside
            # No sign flip needed
            return (1 / (1 + np.exp(self.threshold_ - scores))).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None) -> NDArray[np.str_]:
        """Get feature name for output column."""
        # TODO: what is the mechanism around input_features?
        return np.array([f"{self.feature_name}"])
