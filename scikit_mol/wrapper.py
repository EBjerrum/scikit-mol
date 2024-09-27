"""Wrapper for sklearn estimators and pipelines to handle errors."""

from typing import Any

import numpy as np
import pandas as pd
from functools import wraps
import warnings
from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import available_if
from sklearn.base import TransformerMixin


def filter_invalid_rows(fill_value=np.nan, warn_on_invalid=False):
    def decorator(func):
        @wraps(func)
        def wrapper(obj, X, *args, **kwargs):
            if not getattr(obj, "handle_errors", True):
                # If handle_errors is False, call the original function without filtering
                return func(obj, X, *args, **kwargs)

            valid_mask = np.isfinite(X).all(axis=1)  # Find all rows with nan, inf, etc.

            if warn_on_invalid and not np.all(valid_mask):
                warnings.warn(
                    f"Invalid data detected in {func.__name__}. This may lead to unexpected results.",
                    UserWarning,
                )

            valid_indices = np.where(valid_mask)[0]
            reduced_X = X[valid_mask]

            result = func(obj, reduced_X, *args, **kwargs)

            if result is None:  # For methods like fit that return None
                return None

            if isinstance(result, np.ndarray):
                output = np.full((X.shape[0], result.shape[1]), fill_value)
                output[valid_indices] = result
                return output
            elif isinstance(result, pd.DataFrame):
                # Create a DataFrame with NaN values for all rows
                output = pd.DataFrame(index=range(X.shape[0]), columns=result.columns)
                # Fill the valid rows with the result data
                output.iloc[valid_indices] = result
                return output
            else:
                return result  # For methods that return non-array results

        return wrapper

    return decorator


class NanGuardWrapper(BaseEstimator, TransformerMixin):
    """Nan/Inf safe wrapper for sklearn estimator objects."""

    def __init__(
        self,
        estimator: BaseEstimator,
        handle_errors: bool = False,
        replace_value=np.nan,
    ):
        super().__init__()
        self.handle_errors = handle_errors
        self.replace_value = replace_value
        self.estimator = estimator

    def has_predict(self) -> bool:
        return hasattr(self.estimator, "predict")

    def has_predict_proba(self) -> bool:
        return hasattr(self.estimator, "predict_proba")

    def has_transform(self) -> bool:
        return hasattr(self.estimator, "transform")

    def has_fit_transform(self) -> bool:
        return hasattr(self.estimator, "fit_transform")

    def has_score(self) -> bool:
        return hasattr(self.estimator, "score")

    def has_n_features_in_(self) -> bool:
        return hasattr(self.estimator, "n_features_in_")

    def has_decision_function(self) -> bool:
        return hasattr(self.estimator, "decision_function")

    @property
    def n_features_in_(self) -> int:
        return self.estimator.n_features_in_

    @filter_invalid_rows(warn_on_invalid=True)
    def fit(self, X, *args, **fit_params) -> Any:
        return self.estimator.fit(X, *args, **fit_params)

    @available_if(has_predict)
    @filter_invalid_rows()
    def predict(self, X):
        return self.estimator.predict(X)

    @available_if(has_decision_function)
    @filter_invalid_rows()
    def decision_function(self, X):
        return self.estimator.decision_function(X)

    @available_if(has_predict_proba)
    @filter_invalid_rows()
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    @available_if(has_transform)
    @filter_invalid_rows()
    def transform(self, X):
        return self.estimator.transform(X)

    @available_if(has_fit_transform)
    @filter_invalid_rows(warn_on_invalid=True)
    def fit_transform(self, X, y):
        return self.estimator.fit_transform(X, y)

    @available_if(has_score)
    @filter_invalid_rows(warn_on_invalid=True)
    def score(self, X, y):
        return self.estimator.score(X, y)
