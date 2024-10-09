"""Wrapper for sklearn estimators and pipelines to handle errors."""

from typing import Any

import numpy as np
import pandas as pd
from functools import wraps
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.metaestimators import available_if


class MaskedArrayError(ValueError):
    """Raised when a masked array is passed but handle_errors is False."""

    pass


def filter_invalid_rows(fill_value=np.nan, warn_on_invalid=False):
    def decorator(func):
        @wraps(func)
        def wrapper(obj, X, y=None, *args, **kwargs):
            if not getattr(obj, "safe_inference_mode", True):
                if isinstance(X, np.ma.MaskedArray) and X.mask.any():
                    raise MaskedArrayError(
                        f"Masked array detected with safe_inference_mode=False and {X.mask.any(axis=1).sum()} filtered rows. "
                        "Set safe_inference_mode=True to process masked arrays for inference of production models."
                    )
                return func(obj, X, y, *args, **kwargs)

            # Initialize valid_mask as all True
            valid_mask = np.ones(X.shape[0], dtype=bool)

            # Handle masked arrays
            if isinstance(X, np.ma.MaskedArray):
                valid_mask &= ~X.mask.any(axis=1)

            # Handle non-finite values if required
            if getattr(obj, "mask_nonfinite", True):
                if isinstance(X, np.ma.MaskedArray):
                    valid_mask &= np.isfinite(X.data).all(axis=1)
                else:
                    valid_mask &= np.isfinite(X).all(axis=1)

            if warn_on_invalid and not np.all(valid_mask):
                warnings.warn(
                    f"SafeInferenceWrapper is in safe_inference_mode during use of {func.__name__} and invalid data detected. "
                    "This mode is intended for safe inference in production, not for training and evaluation.",
                    UserWarning,
                )

            valid_indices = np.where(valid_mask)[0]
            reduced_X = X[valid_mask]

            if y is not None:
                reduced_y = y[valid_mask]
            else:
                reduced_y = None

            result = func(obj, reduced_X, reduced_y, *args, **kwargs)

            if result is None:
                return None

            if isinstance(result, np.ndarray):
                if result.ndim == 1:
                    output = np.full(X.shape[0], fill_value)
                else:
                    output = np.full((X.shape[0], result.shape[1]), fill_value)
                output[valid_indices] = result
                return output
            elif isinstance(result, pd.DataFrame):
                output = pd.DataFrame(index=range(X.shape[0]), columns=result.columns)
                output.iloc[valid_indices] = result
                return output
            elif isinstance(result, pd.Series):
                output = pd.Series(index=range(X.shape[0]), dtype=result.dtype)
                output.iloc[valid_indices] = result
                return output
            else:
                return result

        return wrapper

    return decorator


# class NanGuardWrapper(BaseEstimator, TransformerMixin):
#     """Nan/Inf safe wrapper for sklearn estimator objects."""

#     def __init__(
#         self,
#         estimator: BaseEstimator,
#         handle_errors: bool = False,
#         replace_value=np.nan,
#         mask_nonfinite: bool = True,
#     ):
#         super().__init__()
#         self.handle_errors = handle_errors
#         self.replace_value = replace_value
#         self.estimator = estimator
#         self.mask_nonfinite = mask_nonfinite

#     def has_predict(self) -> bool:
#         return hasattr(self.estimator, "predict")

#     def has_predict_proba(self) -> bool:
#         return hasattr(self.estimator, "predict_proba")

#     def has_transform(self) -> bool:
#         return hasattr(self.estimator, "transform")

#     def has_fit_transform(self) -> bool:
#         return hasattr(self.estimator, "fit_transform")

#     def has_score(self) -> bool:
#         return hasattr(self.estimator, "score")

#     def has_n_features_in_(self) -> bool:
#         return hasattr(self.estimator, "n_features_in_")

#     def has_decision_function(self) -> bool:
#         return hasattr(self.estimator, "decision_function")

#     @property
#     def n_features_in_(self) -> int:
#         return self.estimator.n_features_in_

#     @filter_invalid_rows(warn_on_invalid=True)
#     def fit(self, X, *args, **fit_params) -> Any:
#         return self.estimator.fit(X, *args, **fit_params)

#     @available_if(has_predict)
#     @filter_invalid_rows()
#     def predict(self, X):
#         return self.estimator.predict(X)

#     @available_if(has_decision_function)
#     @filter_invalid_rows()
#     def decision_function(self, X):
#         return self.estimator.decision_function(X)

#     @available_if(has_predict_proba)
#     @filter_invalid_rows()
#     def predict_proba(self, X):
#         return self.estimator.predict_proba(X)

#     @available_if(has_transform)
#     @filter_invalid_rows()
#     def transform(self, X):
#         return self.estimator.transform(X)

#     @available_if(has_fit_transform)
#     @filter_invalid_rows(warn_on_invalid=True)
#     def fit_transform(self, X, y):
#         return self.estimator.fit_transform(X, y)

#     @available_if(has_score)
#     @filter_invalid_rows(warn_on_invalid=True)
#     def score(self, X, y):
#         return self.estimator.score(X, y)


class SafeInferenceWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for sklearn estimators to ensure safe inference in production environments.

    This wrapper is designed to be applied to trained models for use in production settings.
    While it can be included during model development and training, the safe inference mode
    should only be enabled when deploying models for inference in production.

    Parameters:
    -----------
    estimator : BaseEstimator
        The trained sklearn estimator to be wrapped.
    safe_inference_mode : bool, default=False
        If True, enables safeguards for handling invalid data during inference.
        This should only be set to True when deploying models to production.
    replace_value : any, default=np.nan
        The value to use for replacing invalid data points.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        safe_inference_mode: bool = True,
        replace_value=np.nan,
        mask_nonfinite: bool = True,
    ):
        self.estimator = estimator
        self.safe_inference_mode = safe_inference_mode
        self.replace_value = replace_value
        self.mask_nonfinite = mask_nonfinite

    @property
    def n_features_in_(self):
        return self.estimator.n_features_in_

    @filter_invalid_rows(warn_on_invalid=True)
    def fit(self, X, y=None, **fit_params):
        return self.estimator.fit(X, y, **fit_params)

    @available_if(lambda self: hasattr(self.estimator, "predict"))
    @filter_invalid_rows()
    def predict(self, X, y=None):
        return self.estimator.predict(X)

    @available_if(lambda self: hasattr(self.estimator, "predict_proba"))
    @filter_invalid_rows()
    def predict_proba(self, X, y=None):
        return self.estimator.predict_proba(X)

    @available_if(lambda self: hasattr(self.estimator, "decision_function"))
    @filter_invalid_rows()
    def decision_function(self, X, y=None):
        return self.estimator.decision_function(X)

    @available_if(lambda self: hasattr(self.estimator, "transform"))
    @filter_invalid_rows()
    def transform(self, X, y=None):
        return self.estimator.transform(X)

    @available_if(lambda self: hasattr(self.estimator, "fit_transform"))
    @filter_invalid_rows(warn_on_invalid=True)
    def fit_transform(self, X, y=None, **fit_params):
        return self.estimator.fit_transform(X, y, **fit_params)

    @available_if(lambda self: hasattr(self.estimator, "score"))
    @filter_invalid_rows(warn_on_invalid=True)
    def score(self, X, y=None):
        return self.estimator.score(X, y)
