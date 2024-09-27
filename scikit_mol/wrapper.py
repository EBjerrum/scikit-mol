"""Wrapper for sklearn estimators and pipelines to handle errors."""

from abc import ABC
from typing import Any

import numpy as np
import pandas as pd
from functools import wraps
import warnings
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import available_if
from sklearn.base import TransformerMixin

from scikit_mol._invalid import (
    rdkit_error_handling,
    # InvalidMol,
    NumpyArrayWithInvalidInstances,
)


class AbstractWrapper(BaseEstimator, ABC):
    """
    Abstract class for the wrapper of sklearn objects.

    Attributes
    ----------
    model: BaseEstimator | Pipeline
        The wrapped model or pipeline.
    """

    model: BaseEstimator | Pipeline

    def __init__(self, replace_invalid: bool, replace_value: Any = np.nan):
        """Initialize the AbstractWrapper.

        Parameters
        ----------
        replace_invalid: bool
            Whether to replace or remove errors
        replace_value: Any, default=np.nan
            If replace_invalid==True, insert this value on the erroneous instance.
        """
        self.replace_invalid = replace_invalid
        self.replace_value = replace_value

    @rdkit_error_handling
    def fit(self, X, y, **fit_params) -> Any:
        return self.model.fit(X, y, **fit_params)

    def has_predict(self) -> bool:
        return hasattr(self.model, "predict")

    def has_fit_predict(self) -> bool:
        return hasattr(self.model, "fit_predict")


class WrappedTransformer(AbstractWrapper):
    """Wrapper for sklearn transformer objects."""

    def __init__(
        self, model: BaseEstimator, replace_invalid: bool = False, replace_value=np.nan
    ):
        """Initialize the WrappedTransformer.

        Parameters
        ----------
        model: BaseEstimator
            Wrapped model to be protected against Errors.
        replace_invalid: bool
            Whether to replace or remove errors
        replace_value: Any, default=np.nan
            If replace_invalid==True, insert this value on the erroneous instance.
        """
        super().__init__(replace_invalid=replace_invalid, replace_value=replace_value)
        self.model = model

    def has_transform(self) -> bool:
        return hasattr(self.model, "transform")

    def has_fit_transform(self) -> bool:
        return hasattr(self.model, "fit_transform")

    @available_if(has_transform)
    @rdkit_error_handling
    def transform(self, X):
        return self.model.transform(X)

    @rdkit_error_handling
    def _fit_transform(self, X, y):
        return self.model.fit_transform(X, y)

    @available_if(has_fit_transform)
    def fit_transform(self, X, y=None):
        out = self._fit_transform(X, y)
        if not self.replace_invalid:
            return out

        if isinstance(out, NumpyArrayWithInvalidInstances):
            return out.array_filled_with(self.replace_value)

        if isinstance(out, list):
            return [self.replace_value if isinstance(v, InvalidMol) else v for v in out]


def filter_invalid_rows(fill_value=np.nan, warn_on_invalid=False):
    def decorator(func):
        @wraps(func)
        def wrapper(obj, X, *args, **kwargs):
            if not getattr(obj, "handle_errors", True):
                # If handle_errors is False, call the original function without filtering
                return func(obj, X, *args, **kwargs)

            valid_mask = np.isfinite(X).all(axis=1)

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
        handle_errors: bool = True,
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
    @filter_invalid_rows()
    def score(self, X, y):
        return self.estimator.score(X, y)
