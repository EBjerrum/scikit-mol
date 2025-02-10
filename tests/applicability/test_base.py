"""Common tests for all applicability domain estimators."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.estimator_checks import check_estimator

# def test_estimator_api(ad_estimator):
#     """Test scikit-learn API compatibility."""
#     check_estimator(ad_estimator)


def test_basic_functionality(ad_estimator, reduced_fingerprints):
    """Test basic fit/transform on reduced fingerprints."""
    ad_estimator.fit(reduced_fingerprints)
    scores = ad_estimator.transform(reduced_fingerprints)
    assert scores.shape == (len(reduced_fingerprints), 1)
    assert np.isfinite(scores).all()


def test_predict_functionality(ad_estimator, ad_test_data):
    """Test predict method returns expected values."""
    X_train, X_test, expected = ad_test_data

    # Fit and predict
    ad_estimator.fit(X_train)
    predictions = ad_estimator.predict(X_test)

    # Check output format
    assert predictions.shape == (len(X_test),)  # Should be 1D
    assert set(np.unique(predictions)) <= {-1, 1}  # Only -1 and 1 allowed

    # Check predictions make sense (in/out of domain)
    accuracy = np.mean(predictions == expected)
    assert accuracy >= 0.8  # Allow some misclassification


def test_score_transform(ad_estimator, ad_test_data):
    """Test score_transform returns valid probability-like scores."""
    X_train, X_test, expected = ad_test_data

    # Fit and get scores
    ad_estimator.fit(X_train)
    scores = ad_estimator.score_transform(X_test)

    # Check output format
    assert scores.shape == (len(X_test), 1)
    assert np.all((0 <= scores) & (scores <= 1))  # Scores in [0,1]

    # Check scores correlate with domain membership
    in_domain = expected == 1
    mean_in = np.mean(scores[in_domain])
    mean_out = np.mean(scores[~in_domain])
    assert mean_in > mean_out  # Inside domain should have higher scores


@pytest.mark.threshold_fitting
def test_threshold_setting(ad_estimator, reduced_fingerprints):
    """Test threshold setting and percentile behavior."""
    if not ad_estimator._supports_threshold_fitting:
        pytest.skip("Estimator does not support threshold fitting")

    # Test default threshold
    ad_estimator.fit(reduced_fingerprints)
    pred_default = ad_estimator.predict(reduced_fingerprints)

    # Test custom percentile
    ad_estimator.percentile = 90
    ad_estimator.fit_threshold(reduced_fingerprints)
    pred_90 = ad_estimator.predict(reduced_fingerprints)

    # More samples should be outside with stricter threshold
    n_inside_default = np.sum(pred_default == 1)
    n_inside_90 = np.sum(pred_90 == 1)
    assert n_inside_90 <= n_inside_default


def test_feature_names(ad_estimator, reduced_fingerprints):
    """Test feature names are properly handled."""
    ad_estimator.fit(reduced_fingerprints)

    # Check feature names exist and match name
    feature_names = ad_estimator.get_feature_names_out()
    assert len(feature_names) == 1
    assert feature_names[0] == ad_estimator.feature_name


def test_pandas_output(ad_estimator, reduced_fingerprints):
    """Test pandas DataFrame output functionality."""
    ad_estimator.set_output(transform="pandas")
    ad_estimator.fit(reduced_fingerprints)

    # Test transform output
    scores_df = ad_estimator.transform(reduced_fingerprints)
    assert hasattr(scores_df, "columns")
    assert len(scores_df.columns) == 1
    assert scores_df.columns[0] == ad_estimator.feature_name

    # Test predict output
    pred_df = ad_estimator.predict(reduced_fingerprints)
    assert hasattr(pred_df, "columns")
    assert len(pred_df.columns) == 1


def test_input_validation(ad_estimator):
    """Test input validation and error handling."""
    # Test fitting with invalid input
    with pytest.raises(ValueError):
        ad_estimator.fit([[]])  # Empty data

    with pytest.raises(ValueError):
        ad_estimator.fit([[1], [2, 3]])  # Inconsistent dimensions

    # Test invalid percentile only if threshold fitting is supported
    if ad_estimator._supports_threshold_fitting:
        with pytest.raises(ValueError):
            ad_estimator.percentile = 101
            ad_estimator.fit([[1, 2]])


def test_refit_consistency(ad_estimator, reduced_fingerprints):
    """Test consistency when refitting with same data."""
    ad_estimator.fit(reduced_fingerprints)
    scores1 = ad_estimator.transform(reduced_fingerprints)

    ad_estimator.fit(reduced_fingerprints)
    scores2 = ad_estimator.transform(reduced_fingerprints)

    assert_array_almost_equal(scores1, scores2)
