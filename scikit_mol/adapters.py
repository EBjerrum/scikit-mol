from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.utils import Bunch
from sklearn.utils._estimator_html_repr import _VisualBlock
from sklearn.utils._user_interface import _print_elapsed_time
from sklearn.utils.metadata_routing import (
    _raise_for_params,
    _routing_enabled,
    process_routing,
)
from sklearn.utils.parallel import Parallel as skParallel
from sklearn.utils.parallel import delayed
from sklearn.utils.validation import check_is_fitted


class EstimatorUnion(FeatureUnion):
    """EXPERIMENTAL: A more flexible version of FeatureUnion that supports various estimator types.

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
                delayed(
                    _transform_one
                )(  # Seems like the only reason we modify this method from base class is to handle it with a custom function for parallel processing
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

    def fit_transform(self, X, y=None, **params):
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        y : array-like of shape (n_samples, n_outputs), default=None
            Targets for supervised learning.

        **params : dict, default=None
            - If `enable_metadata_routing=False` (default):
              Parameters directly passed to the `fit` methods of the
              sub-transformers.

            - If `enable_metadata_routing=True`:
              Parameters safely routed to the `fit` methods of the
              sub-transformers. See :ref:`Metadata Routing User Guide
              <metadata_routing>` for more details.

            .. versionchanged:: 1.5
                `**params` can now be routed via metadata routing API.

        Returns
        -------
        X_t : array-like or sparse matrix of \
                shape (n_samples, sum_n_components)
            The `hstack` of results of transformers. `sum_n_components` is the
            sum of `n_components` (output dimension) over transformers.
        """
        if _routing_enabled():
            routed_params = process_routing(self, "fit_transform", **params)
        else:
            # TODO(SLEP6): remove when metadata routing cannot be disabled.
            routed_params = Bunch()
            for name, obj in self.transformer_list:
                if hasattr(obj, "fit_transform"):
                    routed_params[name] = Bunch(fit_transform={})
                    routed_params[name].fit_transform = params
                else:
                    routed_params[name] = Bunch(fit={})
                    routed_params[name] = Bunch(transform={})
                    routed_params[name].fit = params

        results = self._parallel_func(X, y, _fit_transform_one, routed_params)
        if not results:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        Xs, transformers = zip(*results)
        self._update_transformer_list(transformers)

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


def _fit_transform_one(
    transformer,
    X,
    y,
    weight,
    message_clsname="",
    message=None,
    params=None,
    method="transform",
):
    """
    Fits ``transformer`` to ``X`` and ``y``. The transformed result is returned
    with the fitted transformer. If ``weight`` is not ``None``, the result will
    be multiplied by ``weight``.

    ``params`` needs to be of the form ``process_routing()["step_name"]``.
    """
    params = params or {}
    with _print_elapsed_time(message_clsname, message):
        if hasattr(transformer, "fit_transform"):
            res = transformer.fit_transform(X, y, **params.get("fit_transform", {}))
        elif hasattr(transformer, "transform"):
            res = transformer.fit(X, y, **params.get("fit", {})).transform(
                X, **params.get("transform", {})
            )
        # Custom handling of methods that has predict but no fit_transform or transform
        elif hasattr(transformer, "predict"):
            transformer.fit(X, y, **params.get("fit", {}))
            res = transformer.predict(X, **params.get("predict", {}))
            if res.ndim == 1:
                res = res.reshape(-1, 1)
        else:
            raise ValueError(
                f"Transformer {transformer} does not have a fit_transform, fit or predict method."
            )

    if weight is None:
        return res, transformer
    return res * weight, transformer


class _BaseAdapter(BaseEstimator):
    """EXPERIMENTAL: Base class for adapters that wrap estimators and modify their interface."""

    def __init__(
        self, estimator: BaseEstimator, _feature_names_out: Optional[List[str]] = None
    ):
        """Initialize the adapter with an estimator."""
        self.estimator = estimator
        self._feature_names_out = _feature_names_out

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features."""

        return ["tester"]

    def __getattr__(self, name):
        """Delegate any unknown attributes/methods to wrapped estimator."""
        if hasattr(self.estimator, name):
            attr = getattr(self.estimator, name)
            if isinstance(attr, property):
                return attr.__get__(self.estimator)
            return attr
        raise AttributeError(
            f"Neither {self.__class__.__name__} nor {self.estimator.__class__.__name__} "
            f"has attribute '{name}'"
        )

    def __dir__(self):
        """List all attributes including those from wrapped estimator."""
        return list(set(super().__dir__() + dir(self.estimator)))

    @property
    def __dict__(self):
        """Include estimator's properties in the instance dict."""
        d = super().__dict__.copy()
        estimator_dict = vars(self.estimator)
        for name, value in estimator_dict.items():
            if not name.startswith("_"):
                d[name] = value
        return d

    def _sk_visual_block_(self):
        # TODO: this looks strange when putting the wrapped estimator into a pipeline
        """Generate information about how to display the adapter."""
        return _VisualBlock(
            "parallel",
            [self.estimator],
            names=None,
            name_details=None,
            name_caption=None,
            dash_wrapped=False,
        )


class PredictToTransformAdapter(_BaseAdapter, TransformerMixin):
    """EXPERIMENTAL: Adapter that exposes an estimator's predict method as transform."""

    def __init__(self, estimator: BaseEstimator, method: str = "predict"):
        """Initialize the adapter with an estimator and a method to use.

        Parameters
        ----------
        estimator : BaseEstimator
            The estimator to wrap.
        method : str, default="predict"
            The method to use for transformation.
        """
        super().__init__(estimator)
        self.method = method

    def transform(self, X):
        """Transform X using the wrapped estimator's specified method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to transform.
        check_is_fitted(self)

        Example
        --------
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn_mol.adapters import PredictToTransformAdapter
        >>> estimator = LogisticRegression()
        >>> adapter = PredictToTransformAdapter(estimator, method="predict"")
        >>> adapter.fit(X, y)
        >>> adapter.transform(X)
        """
        prediction = getattr(self.estimator, self.method)(X)
        if prediction.ndim == 1:
            prediction = prediction.reshape(-1, 1)
        return prediction


class TransformToPredictAdapter(_BaseAdapter, TransformerMixin):
    """EXPERIMENTAL: Adapter that exposes an estimator's transform method as predict.

    2D column vector output is flattened to 1D."""

    def __init__(self, estimator: BaseEstimator, method: str = "transform"):
        super().__init__(estimator)
        self.method = method

    def predict(self, X):
        """Predict using the wrapped estimator's specified method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to predict.

        Example
        --------
        >>> from sklearn.preprocessing import StandardScaler
        >>> from sklearn_mol.adapters import TransformToPredictAdapter
        >>> estimator = StandardScaler()
        >>> adapter = TransformToPredictAdapter(estimator, method="transform")
        >>> adapter.fit(X, y)
        >>> adapter.predict(X)
        """
        check_is_fitted(self)
        prediction = self.estimator.transform(X)
        if prediction.shape[1] == 1:
            prediction = prediction.flatten()
        return prediction
