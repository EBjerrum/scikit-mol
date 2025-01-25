"""Wrapper for sklearn estimators and pipelines to handle errors."""

import warnings
from functools import wraps
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import NotFittedError, check_is_fitted

from .utilities import set_safe_inference_mode

__all__ = ["MaskedArrayError", "SafeInferenceWrapper", "set_safe_inference_mode"]


class MaskedArrayError(ValueError):
    """Raised when a masked array is passed but safe_inference_mode is False."""

    pass


def filter_invalid_rows(warn_on_invalid=False, replace_value=np.nan):
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
            if not hasattr(obj, "replace_value"):
                raise ValueError(
                    "replace_value must be set in the SafeInferenceWrapper"
                )
            else:
                replace_value = obj.replace_value

            # Initialize valid_mask as all True
            valid_mask = np.ones(X.shape[0], dtype=bool)

            # Handle masked arrays
            if isinstance(X, np.ma.MaskedArray):
                try:
                    valid_mask &= ~X.mask.any(axis=1)
                # workaround for situation where mask is single boolean (all masked/ all unmasked) and no axis present
                except np.exceptions.AxisError:
                    valid_mask &= ~X.mask

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
                # TODO, how can we check y in the same way as the estimator?
                y = check_array(
                    y,
                    force_all_finite=False,  # accept_sparse="csr",
                    ensure_2d=False,
                    dtype=None,
                    input_name="y",
                    estimator=obj,
                )
                reduced_y = y[valid_mask]
            else:
                reduced_y = None

            # handle case where all rows are masked e.g single invalid input is passed
            if len(valid_indices) == 0:
                result = np.array([])
            else:
                result = func(obj, reduced_X, reduced_y, *args, **kwargs)

            if result is None:
                return None
            if isinstance(result, np.ndarray):
                if result.ndim == 1:
                    output = np.full(X.shape[0], replace_value)
                else:
                    output = np.full((X.shape[0], result.shape[1]), replace_value)
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


class SafeInferenceWrapper(TransformerMixin, BaseEstimator):
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
        safe_inference_mode: bool = False,
        replace_value: Union[int, float, str] = np.nan,
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

    @available_if(lambda self: hasattr(self.estimator, "get_feature_names_out"))
    @filter_invalid_rows(warn_on_invalid=True)
    def get_feature_names_out(self, *args, **kwargs):
        return self.estimator.get_feature_names_out(*args, **kwargs)

    def __sklearn_is_fitted__(self):
        try:
            check_is_fitted(self.estimator)
            return True
        except NotFittedError:
            return False
