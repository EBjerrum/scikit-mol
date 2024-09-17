"""Wrapper for sklearn estimators and pipelines to handle errors."""

from abc import ABC
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import available_if

from scikit_mol._invalid import rdkit_error_handling, InvalidInstance, NumpyArrayWithInvalidInstances


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

    def __init__(self, model: BaseEstimator, replace_invalid: bool = False, replace_value=np.nan):
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
        out = self._fit_transform(X,y)
        if not self.replace_invalid:
            return out

        if isinstance(out, NumpyArrayWithInvalidInstances):
            return out.array_filled_with(self.replace_value)

        if isinstance(out, list):
            return [self.replace_value if isinstance(v, InvalidInstance) else v for v in out]



