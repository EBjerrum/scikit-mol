from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.utils import Bunch
from sklearn.utils.metadata_routing import (
    _raise_for_params,
    _routing_enabled,
    process_routing,
)
from sklearn.utils.parallel import Parallel as skParallel
from sklearn.utils.parallel import delayed
from sklearn.utils.validation import check_is_fitted


class EstimatorUnion(FeatureUnion):
    """EXPERIMENTAL: more flexible version of FeatureUnion that supports various estimator types.

    This class extends scikit-learn's FeatureUnion to support estimators with different
    method interfaces (predict, transform, etc.) and allows explicit method selection.
    It maintains all functionality of FeatureUnion while adding flexible method resolution.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        List of (name, estimator) tuples, where estimator is any scikit-learn
        compatible estimator with at least one of the methods specified in
        method_resolution_order.
    method_resolution_order : tuple of str, default=("predict", "transform")
        Ordered tuple of method names to try when getting output from estimators.
        Methods are tried in order until a valid one is found.
    selected_methods : dict or None, default=None
        Optional mapping of estimator names to specific methods to use. Takes
        precedence over method_resolution_order.
    n_jobs : int or None, default=None
        Number of jobs to run in parallel. None means 1.
    transformer_weights : dict or None, default=None
        Multiplicative weights for features per transformer. Keys are transformer
        names, values are weights.
    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be printed.
    verbose_feature_names_out : bool, default=True
        If True, the feature names out will be verbose.

    Attributes
    ----------
    transformers_ : list
        List of fitted transformers.

    Notes
    -----
    This class inherits from FeatureUnion and maintains all its functionality including
    parallel processing, transformer weights, and metadata routing. The key extension is
    the ability to handle estimators with different method interfaces through configurable
    method resolution.

    See Also
    --------
    sklearn.pipeline.FeatureUnion : The parent class providing base functionality.
    """

    def __init__(
        self,
        estimator_list: List[Tuple[str, BaseEstimator]],
        *,
        method_resolution_order: Tuple[str, ...] = ("predict", "transform"),
        selected_methods: Optional[Dict[str, str]] = None,
        n_jobs: Optional[int] = None,
        transformer_weights: Optional[Dict[str, float]] = None,
        verbose: bool = False,
        verbose_feature_names_out: bool = True,
    ) -> None:
        # Store all parameters as properties
        self.estimator_list = estimator_list
        self.method_resolution_order = method_resolution_order
        self.selected_methods = selected_methods or {}
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose
        self.verbose_feature_names_out = verbose_feature_names_out

    @property
    def estimator_list(self) -> List[Tuple[str, BaseEstimator]]:
        """Get estimators (alias for transformer_list)."""
        return self.transformer_list

    @estimator_list.setter
    def estimator_list(self, estimator_list: List[Tuple[str, BaseEstimator]]) -> None:
        """Set estimators (and internal transformer_list property)."""
        self.transformer_list = estimator_list

    def _get_method_name(self, estimator_tuple: Tuple[str, BaseEstimator]) -> str:
        """Get the appropriate method name for the estimator, raising clear errors if not found.

        Parameters
        ----------
        estimator_tuple : tuple of (str, estimator)
            Tuple containing the estimator name and instance.

        Returns
        -------
        str
            Name of the method to use for this estimator.

        Raises
        ------
        ValueError
            If no valid method is found for the estimator.
        """
        name, estimator = estimator_tuple

        # Check explicit method if specified
        if name in self.selected_methods:
            method = self.selected_methods[name]
            if not hasattr(estimator, method):
                raise ValueError(
                    f"Estimator '{name}' ({type(estimator).__name__}) does not have "
                    f"explicitly selected method '{method}'. Consider changing selected_methods "
                    f"or using only method_resolution_order to specify valid methods."
                )
            return method

        # Try methods in resolution order
        for method in self.method_resolution_order:
            if hasattr(estimator, method):
                return method

        raise ValueError(
            f"Estimator '{name}' ({type(estimator).__name__}) does not have any of "
            f"the methods: {', '.join(self.method_resolution_order)}. Consider using "
            f"method_resolution_order or selected_methods to specify valid methods."
        )

    def _get_estimator_output(
        self, estimator_tuple: Tuple[str, BaseEstimator], X: NDArray
    ) -> NDArray:
        """Get output from estimator using appropriate method."""
        name, estimator = estimator_tuple
        method = self._get_method_name(estimator_tuple)
        output = getattr(estimator, method)(X)

        # Ensure 2D output
        if output.ndim == 1:
            output = output.reshape(-1, 1)
        return output

    # def transform_old(self, X: NDArray) -> NDArray:
    #     """Transform X using the selected method for each estimator.

    #     Parameters
    #     ----------
    #     X : array-like of shape (n_samples, n_features)
    #         Input data to be transformed.

    #     Returns
    #     -------
    #     ndarray of shape (n_samples, sum_n_output_features)
    #         Horizontally stacked results of all estimators.
    #         sum_n_output_features is the sum of n_output_features for each
    #         estimator.
    #     """
    #     Xs = self._parallel_func(X, self._get_estimator_output)
    #     if not Xs:
    #         # All transformers are None
    #         return np.zeros((X.shape[0], 0))

    #     if self.transformer_weights is not None:
    #         Xs = [
    #             (Xs[name] * self.transformer_weights[name])
    #             if name in self.transformer_weights
    #             else Xs[name]
    #             for name in self._iter()
    #         ]

    #     return np.hstack(Xs)
    def _validate_transformers(self):
        names, transformers = zip(*self.transformer_list)

        # validate names
        self._validate_names(names)

        # validate estimators
        for t in transformers:
            if t in ("drop", "passthrough"):
                continue
            # TODO, make a check that the methods in the method_resolution_order /method mappting are present
            # if not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not hasattr(
            #     t, "transform"
            # ):
            #     raise TypeError(
            #         "All estimators should implement fit and "
            #         "transform. '%s' (type %s) doesn't" % (t, type(t))
            #     )

    def transform(self, X, **params):
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        **params : dict, default=None

            Parameters routed to the `transform` method of the sub-transformers via the
            metadata routing API. See :ref:`Metadata Routing User Guide
            <metadata_routing>` for more details.

            .. versionadded:: 1.5

        Returns
        -------
        X_t : array-like or sparse matrix of shape (n_samples, sum_n_components)
            The `hstack` of results of transformers. `sum_n_components` is the
            sum of `n_components` (output dimension) over transformers.
        """
        _raise_for_params(params, self, "transform")

        if _routing_enabled():
            routed_params = process_routing(self, "transform", **params)
        else:
            # TODO(SLEP6): remove when metadata routing cannot be disabled.
            routed_params = Bunch()
            for name, _ in self.transformer_list:
                routed_params[name] = Bunch(transform={})

        # Build delayed jobs with custom methods
        delayed_jobs = []
        for name, trans, weight in self._iter():
            method_name = self._get_method_name((name, trans))
            delayed_jobs.append(
                delayed(_transform_one)(
                    trans,
                    X,
                    None,
                    weight,
                    params=routed_params[name],
                    method=method_name,
                )
            )

        Xs = skParallel(n_jobs=self.n_jobs)(delayed_jobs)

        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        return self._hstack(Xs)

    def predict(self, X: NDArray) -> NDArray:
        """Predict using all estimators.

        Alias for transform to maintain predictor interface.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to be predicted.

        Returns
        -------
        ndarray of shape (n_samples, sum_n_output_features)
            Horizontally stacked predictions of all estimators.
        """
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        # List of tuples (name, feature_names_out)
        transformer_with_feature_names_out = []
        for name, trans, _ in self._iter():
            if hasattr(trans, "predict") and not hasattr(
                trans, "get_feature_names_out"
            ):
                # Assume predictors only return 1D output and thus we use their name as feature_name
                feature_names_out = np.array([name])
            elif not hasattr(trans, "get_feature_names_out"):
                raise AttributeError(
                    "Transformer %s (type %s) does not provide get_feature_names_out."
                    % (str(name), type(trans).__name__)
                )
            else:
                feature_names_out = trans.get_feature_names_out(input_features)
            transformer_with_feature_names_out.append((name, feature_names_out))

        return self._add_prefix_for_feature_names_out(
            transformer_with_feature_names_out
        )


def _transform_one(transformer, X, y, weight, params=None, method="transform"):
    """Call transform and apply weight to output.

    Parameters
    ----------
    transformer : estimator
        Estimator to be used for transformation.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input data to be transformed.

    y : ndarray of shape (n_samples,)
        Ignored.

    weight : float
        Weight to be applied to the output of the transformation.

    method : str
        Method to use for transformation (e.g. "transform", "predict", "predict_proba").

    params : dict
        Parameters to be passed to the transformer's ``transform`` method.

        This should be of the form ``process_routing()["step_name"]``.
    """
    res = getattr(transformer, method)(X, **params.transform)
    # Ensure 2D output
    if res.ndim == 1:
        res = res.reshape(-1, 1)
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res
    return res * weight


# def _fit_transform_one(
#     transformer, X, y, weight, message_clsname="", message=None, params=None, method="transform"
# ):
#     """
#     Fits ``transformer`` to ``X`` and ``y``. The transformed result is returned
#     with the fitted transformer. If ``weight`` is not ``None``, the result will
#     be multiplied by ``weight``.

#     ``params`` needs to be of the form ``process_routing()["step_name"]``.
#     """
#     params = params or {}
#     with _print_elapsed_time(message_clsname, message):
#         if hasattr(transformer, "fit_transform"):
#             res = transformer.fit_transform(X, y, **params.get("fit_transform", {}))
#         else:
#             res = transformer.fit(X, y, **params.get("fit", {})).transform(
#                 X, **params.get("transform", {})
#             )

#     if weight is None:
#         return res, transformer
#     return res * weight, transformer


class PredictToTransformAdapter(TransformerMixin, BaseEstimator):
    """Adapter that exposes an estimator's predict method as transform.

    Parameters
    ----------
    estimator : BaseEstimator
        Estimator with a predict method.
    method : str, default="predict"
        The method to use for transformation (e.g., "predict", "predict_proba").
    """

    def __init__(self, estimator: BaseEstimator, method: str = "predict"):
        self.estimator = estimator
        self.method = method

    def fit(self, X, y=None):
        self.estimator.fit(X, y)
        return self

    def transform(self, X):
        check_is_fitted(self)
        return getattr(self.estimator, self.method)(X)

    def get_feature_names_out(self, input_features=None):
        """Delegate feature names to wrapped estimator if available."""
        if hasattr(self.estimator, "get_feature_names_out"):
            return self.estimator.get_feature_names_out(input_features)
        return None

    def __sklearn_is_fitted__(self):
        """Delegate fit check to wrapped estimator."""
        try:
            check_is_fitted(self.estimator)
            return True
        except ValueError:
            return False

    def _repr_html_(self):
        """HTML representation for notebooks."""
        if hasattr(self.estimator, "_repr_html_"):
            return f"<div>PredictToTransformAdapter using method '{self.method}' on:<br/>{self.estimator._repr_html_()}</div>"
        return f"<div>PredictToTransformAdapter(method='{self.method}', estimator={self.estimator})</div>"


class TransformToPredictAdapter(BaseEstimator):
    """Adapter that exposes an estimator's transform method as predict.

    Parameters
    ----------
    transformer : BaseEstimator
        Estimator with a transform method.
    """

    def __init__(self, transformer: BaseEstimator):
        self.transformer = transformer

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.transformer.transform(X)

    def get_feature_names_out(self, input_features=None):
        """Delegate feature names to wrapped transformer if available."""
        if hasattr(self.transformer, "get_feature_names_out"):
            return self.transformer.get_feature_names_out(input_features)
        return None

    def __sklearn_is_fitted__(self):
        """Delegate fit check to wrapped transformer."""
        try:
            check_is_fitted(self.transformer)
            return True
        except ValueError:
            return False

    def _repr_html_(self):
        """HTML representation for notebooks."""
        if hasattr(self.transformer, "_repr_html_"):
            return f"<div>TransformToPredictAdapter on:<br/>{self.transformer._repr_html_()}</div>"
        return f"<div>TransformToPredictAdapter(transformer={self.transformer})</div>"
