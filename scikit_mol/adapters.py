import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils._pprint import _EstimatorPrettyPrinter
from sklearn.utils._param_validation import validate_params
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.utils._estimator_html_repr import _VisualBlock
from sklearn.utils._set_output import _safe_set_output
from scipy.stats import chi2


class EstimatorUnion(BaseEstimator, TransformerMixin):
    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y=None):
        self.fitted_estimators_ = []
        for name, estimator in self.estimators:
            if hasattr(estimator, "fit"):
                fitted_estimator = estimator.fit(X, y)
                self.fitted_estimators_.append((name, fitted_estimator))
            else:
                self.fitted_estimators_.append((name, estimator))
        return self

    def transform(self, X):
        check_is_fitted(self)
        results = []
        for name, estimator in self.fitted_estimators_:
            if hasattr(estimator, "predict"):
                results.append(estimator.predict(X))
            elif hasattr(estimator, "transform"):
                results.append(estimator.transform(X))
        return np.column_stack(results)

    def predict(self, X):
        return self.transform(X)

    @available_if(lambda self: hasattr(self, "fitted_estimators_"))
    def get_feature_names_out(self, input_features=None):
        feature_names = []
        for name, estimator in self.fitted_estimators_:
            if hasattr(estimator, "get_feature_names_out"):
                feature_names.extend(estimator.get_feature_names_out())
            else:
                feature_names.append(name)
        return np.array(feature_names)

    def set_output(self, *, transform=None):
        """Set output container for all estimators.

        Parameters
        ----------
        transform : {"default", "pandas"}, default=None
            Configure output of `transform` and `fit_transform`.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        for _, estimator in self.estimators:
            _safe_set_output(estimator, transform=transform)
        return super().set_output(transform=transform)

    def __repr__(self):
        class_name = self.__class__.__name__
        estimator_reprs = []
        for name, estimator in self.estimators:
            estimator_repr = f"{name}={estimator.__repr__()}"
            estimator_reprs.append(estimator_repr)
        estimators_str = ",\n".join(estimator_reprs)
        return f"{class_name}([\n{estimators_str}\n])"

    def _sk_visual_block_(self):
        names, transformers = zip(*self.estimators)
        return _VisualBlock("parallel", transformers, names=names)


class SigmoidThresholdTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold, steepness=1, feature_name="Sigmoid_", prefix=True):
        self.threshold = threshold
        self.steepness = steepness
        self.feature_name = feature_name
        self.prefix = prefix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return 1 / (1 + np.exp(self.steepness * (X - self.threshold)))

    def predict(self, X):
        return self.transform(X)

    @available_if(lambda self: hasattr(self, "fitted_estimators_"))
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)

        if input_features is None:
            if (
                hasattr(self, "feature_names_in_")
                and self.feature_names_in_ is not None
            ):
                input_features = self.feature_names_in_
            else:
                input_features = [f"x{i}" for i in range(self.n_features_in_)]

        if self.feature_name:
            if self.prefix:
                return np.array(
                    [f"{self.feature_name}{feature}" for feature in input_features]
                )
            else:
                if len(input_features) > 1:
                    return np.array(
                        [f"{self.feature_name}{i}" for i in range(len(input_features))]
                    )
                else:
                    return np.array([self.feature_name])
        else:
            return np.array(input_features)


class NullEstimator(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    def __init__(
        self,
        accept_sparse=False,
    ):
        self.accept_sparse = accept_sparse

    def fit(self, X, y=None):
        # Check and store the input
        self.X_ = check_array(
            X, accept_sparse=self.accept_sparse, force_all_finite="allow-nan"
        )
        self.n_features_in_ = self.X_.shape[1]
        self.feature_names_in_ = getattr(X, "columns", None)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(
            X, accept_sparse=self.accept_sparse, force_all_finite="allow-nan"
        )

        # Check that the input is of the same shape as the one passed during fit.
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Shape of input is different from what was seen in `fit`"
                f" Expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        return X

    def predict(self, X):
        return self.transform(X)

    @available_if(lambda self: hasattr(self, "fitted_estimators_"))
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)

        # Do I need to heck that the size of input_features is correct?
        # if len(input_features) != self.n_features_out_:
        #     raise ValueError(f"Expected {self.n_features_in_} features, got {len(input_features)}")

        if input_features:
            return input_features
        else:
            return np.array([f"x{i}" for i in range(self.n_features_in_)])

    def _more_tags(self):
        return {
            "allow_nan": True,
            "X_types": ["2darray"] + (["sparse"] if self.accept_sparse else []),
        }
