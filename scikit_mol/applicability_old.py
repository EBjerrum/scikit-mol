import numpy as np
from scipy import linalg, stats
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin, check_array, check_is_fitted
from sklearn.neighbors import NearestNeighbors


class NearestNeighborsDistance(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors
        self.feature_name = "nn_distance"

    def fit(self, X, y=None):
        self.X_sparse = csr_matrix(X)
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="cosine")
        self.nn.fit(self.X_sparse)
        return self

    def transform(self, X):
        X_sparse = csr_matrix(X)
        distances, _ = self.nn.kneighbors(X_sparse)
        avg_distances = np.mean(distances, axis=1)
        return avg_distances.reshape(-1, 1)  # Return 2D array for consistency

    def predict(self, X):
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.array([self.feature_name])


class LeverageDistanceSlow(BaseEstimator, TransformerMixin):
    """Calculate leverage-based distances for applicability domain assessment.

    The leverage approach measures how far a sample is from the center of the
    X variable space. It's based on the hat matrix H = X(X'X)^(-1)X'.

    Parameters
    ----------
    threshold_factor : float, default=3
        Factor used in calculating the leverage threshold h* = threshold_factor * (p+1)/n
        where p is the number of features and n is the number of samples.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    X_fit_ : ndarray
        Training data used in fit.
    leverage_threshold_ : float
        Calculated leverage threshold (h*).
    """

    def __init__(self, threshold_factor=3):
        self.threshold_factor = threshold_factor

    def fit(self, X, y=None):
        """Fit the model using X as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = check_array(X, accept_sparse=False)
        self.n_features_in_ = X.shape[1]
        self.X_fit_ = X

        # Calculate leverage threshold h*
        n_samples = X.shape[0]
        self.leverage_threshold_ = (
            self.threshold_factor * (self.n_features_in_ + 1) / n_samples
        )

        # Store (X'X)^(-1) for later use
        self.xtx_inv_ = np.linalg.inv(X.T @ X)

        return self

    def transform(self, X):
        """Calculate leverage-based distances for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to calculate leverage distances for.

        Returns
        -------
        h : ndarray of shape (n_samples, 1)
            The leverage values for each sample.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but LeverageDistance "
                f"was fitted with {self.n_features_in_} features."
            )

        # Calculate leverage values h = diag(X(X'X)^(-1)X')
        # Slighlty different implementation (from another package)
        # hat_matrix = X @ self.xtx_inv_ @ X.T
        # leverages = np.diag(hat_matrix)

        h = np.sum(X @ self.xtx_inv_ * X, axis=1)

        return h.reshape(-1, 1)

    def predict(self, X):
        """Alias for transform, following scikit-learn conventions."""
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names.

        Parameters
        ----------
        input_features : None
            Ignored as the transformer generates new feature names.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Leverage distance feature name.
        """
        check_is_fitted(self)
        return np.array(["leverage_distance"])


# Faster but gives some _very_ large distances for some compounds!
class LeverageDistance(BaseEstimator, TransformerMixin):
    """Calculate leverage-based distances for applicability domain assessment.

    Parameters
    ----------
    threshold_factor : float, default=3
        Factor used in calculating the leverage threshold h* = threshold_factor * (p+1)/n
    """

    def __init__(self, threshold_factor=3):
        self.threshold_factor = threshold_factor

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=False)
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]

        # Calculate leverage threshold h*
        self.leverage_threshold_ = (
            self.threshold_factor * (self.n_features_in_ + 1) / n_samples
        )

        # Use more efficient matrix operations
        # Calculate (X'X)^(-1) using SVD which is more stable
        U, s, Vh = linalg.svd(X, full_matrices=False)

        # Store components for faster transform
        self.s_inv_ = 1 / s
        self.U_ = U
        self.Vh_ = Vh

        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but LeverageDistance "
                f"was fitted with {self.n_features_in_} features."
            )

        # Efficient leverage calculation using stored SVD components
        # This avoids explicit matrix inversion
        Z = X @ self.Vh_.T * self.s_inv_
        h = np.sum(Z * Z, axis=1)

        return h.reshape(-1, 1)

    def predict(self, X):
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return np.array(["leverage_distance"])


class MahalanobisDistance(BaseEstimator, TransformerMixin):
    """Calculate Mahalanobis distances for applicability domain assessment.

    Parameters
    ----------
    threshold_quantile : float, default=0.975
        Quantile of chi-square distribution to use as threshold.
    threshold_strategy : str, default='chi2'
        Strategy to compute threshold. Options:
        - 'chi2': Use chi-square distribution (theoretical)
        - 'empirical': Use empirical distribution from training data
        - None: Don't compute threshold (useful for CV)
    """

    def __init__(self, threshold_quantile=0.975, threshold_strategy="chi2"):
        self.threshold_quantile = threshold_quantile
        self.threshold_strategy = threshold_strategy

    def fit(self, X, y=None):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]

        # Compute mean and covariance
        self.mean_ = np.mean(X, axis=0)
        self.covariance_ = np.cov(X, rowvar=False)
        self.inv_covariance_ = np.linalg.inv(self.covariance_)

        # Calculate distances for training set
        train_distances = self._mahalanobis(X)
        self.train_distances_ = train_distances

        # Set threshold based on strategy
        if self.threshold_strategy == "chi2":
            self.threshold_ = stats.chi2.ppf(
                self.threshold_quantile, df=self.n_features_in_
            )
        elif self.threshold_strategy == "empirical":
            self.threshold_ = np.quantile(train_distances, self.threshold_quantile)
        elif self.threshold_strategy is None:
            self.threshold_ = None
        else:
            raise ValueError(f"Unknown threshold_strategy: {self.threshold_strategy}")

        return self

    def _mahalanobis(self, X):
        """Calculate Mahalanobis distances."""
        X_centered = X - self.mean_
        return np.sqrt(np.sum(X_centered @ self.inv_covariance_ * X_centered, axis=1))

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {self.__class__.__name__} "
                f"was fitted with {self.n_features_in_} features."
            )

        distances = self._mahalanobis(X)
        return distances.reshape(-1, 1)

    def set_threshold(self, threshold):
        """Set threshold manually, e.g., from cross-validation."""
        self.threshold_ = threshold
        return self

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return np.array(["mahalanobis_distance"])

    def _more_tags(self):
        return {
            "requires_fit": True,
            "X_types": ["2darray"],
            "poor_score": False,
            "allow_nan": False,
        }
