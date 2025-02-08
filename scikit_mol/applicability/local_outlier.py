"""
Local Outlier Factor applicability domain.

This module was adapted from MLChemAD (https://github.com/OlivierBeq/MLChemAD)
Original work Copyright (c) 2023 Olivier J. M. Béquignon (MIT License)
Modifications Copyright (c) 2025 scikit-mol contributors (LGPL License)
See LICENSE.MIT in this directory for the original MIT license.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils.validation import check_array, check_is_fitted


class LocalOutlierFactorApplicabilityDomain(BaseEstimator, TransformerMixin):
    """Applicability domain based on Local Outlier Factor (LOF).

    LOF measures the local deviation of density of a sample with respect to its
    neighbors, identifying samples that have substantially lower density than
    their neighbors.

    Parameters
    ----------
    n_neighbors : int, default=20
        Number of neighbors to use for LOF calculation.
    contamination : float, default=0.1
        Expected proportion of outliers in the data set.
    metric : str, default='euclidean'
        Metric to use for distance computation.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    lof_ : LocalOutlierFactor
        Fitted LOF estimator.

    Examples
    --------
    >>> from scikit_mol.applicability import LocalOutlierFactorApplicabilityDomain
    >>> ad = LocalOutlierFactorApplicabilityDomain()
    >>> ad.fit(X_train)
    >>> predictions = ad.predict(X_test)

    References
    ----------
    .. [1] Breunig et al. (2000). LOF: Identifying Density-Based Local Outliers.
           In: Proc. 2000 ACM SIGMOD Int. Conf. Manag. Data, ACM, pp. 93-104.
    """

    def __init__(self, n_neighbors=20, contamination=0.1, metric="euclidean"):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric

    def fit(self, X, y=None):
        """Fit the LOF applicability domain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = check_array(X)
        self.n_features_in_ = X.shape[1]

        self.lof_ = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            contamination=self.contamination,
            novelty=True,
        )
        self.lof_.fit(X)

        return self

    def transform(self, X):
        """Calculate LOF scores for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        scores : ndarray of shape (n_samples, 1)
            The LOF scores of the samples. Higher scores indicate samples
            that are more likely to be outliers.
        """
        check_is_fitted(self)
        X = check_array(X)

        # Get negative LOF scores (higher means more likely to be inlier)
        scores = -self.lof_.score_samples(X)
        return scores.reshape(-1, 1)

    def predict(self, X):
        """Predict whether samples are within the applicability domain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Returns 1 for samples inside the domain and -1 for samples outside
            (following scikit-learn's convention for outlier detection).
        """
        return self.lof_.predict(X)
