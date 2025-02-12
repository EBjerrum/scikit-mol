from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.pipeline import FeatureUnion


class EstimatorUnion(FeatureUnion):
    """A more flexible version of FeatureUnion that supports various estimator types.

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
    ) -> None:
        # Store all parameters as properties
        self.estimator_list = estimator_list
        self.method_resolution_order = method_resolution_order
        self.selected_methods = selected_methods or {}
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose

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

    def transform(self, X: NDArray) -> NDArray:
        """Transform X using the selected method for each estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to be transformed.

        Returns
        -------
        ndarray of shape (n_samples, sum_n_output_features)
            Horizontally stacked results of all estimators.
            sum_n_output_features is the sum of n_output_features for each
            estimator.
        """
        Xs = self._parallel_func(X, self._get_estimator_output)
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        if self.transformer_weights is not None:
            Xs = [
                (Xs[name] * self.transformer_weights[name])
                if name in self.transformer_weights
                else Xs[name]
                for name in self._iter()
            ]

        return np.hstack(Xs)

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
