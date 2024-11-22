import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.fingerprints import MorganFingerprintTransformer
from scikit_mol.safeinference import SafeInferenceWrapper
from scikit_mol.utilities import set_safe_inference_mode

from fixtures import (
    SLC6A4_subset,
    invalid_smiles_list,
    skip_pandas_output_test,
    smiles_list,
)


@pytest.fixture
def smiles_pipeline():
    return Pipeline(
        [
            ("s2m", SmilesToMolTransformer()),
            ("FP", MorganFingerprintTransformer()),
            (
                "RF",
                SafeInferenceWrapper(
                    RandomForestRegressor(n_estimators=3, random_state=42)
                ),
            ),
        ]
    )


def test_safeinference_wrapper_basic(smiles_pipeline, SLC6A4_subset):
    X_smiles, Y = SLC6A4_subset.SMILES, SLC6A4_subset.pXC50
    X_smiles = X_smiles.to_frame()

    # Set safe inference mode
    set_safe_inference_mode(smiles_pipeline, True)

    # Train the model
    smiles_pipeline.fit(X_smiles, Y)

    # Test prediction
    predictions = smiles_pipeline.predict(X_smiles)
    assert len(predictions) == len(X_smiles)
    assert not np.any(np.isnan(predictions))


def test_safeinference_wrapper_with_invalid_smiles(
    smiles_pipeline, SLC6A4_subset, invalid_smiles_list
):
    X_smiles, Y = SLC6A4_subset.SMILES[:100], SLC6A4_subset.pXC50[:100]
    X_smiles = X_smiles.to_frame()

    # Set safe inference mode
    set_safe_inference_mode(smiles_pipeline, True)

    # Train the model
    smiles_pipeline.fit(X_smiles, Y)

    # Create a test set with invalid SMILES
    X_test = pd.DataFrame({"SMILES": X_smiles["SMILES"].tolist() + invalid_smiles_list})

    # Test prediction with invalid SMILES
    predictions = smiles_pipeline.predict(X_test)
    assert len(predictions) == len(X_test)
    assert np.any(np.isnan(predictions))
    assert np.all(np.isnan(predictions[-1]))  # Only last should be nan
    assert np.all(~np.isnan(predictions[:-1]))  # All others should not be nan


def test_safeinference_wrapper_without_safe_mode(
    smiles_pipeline, SLC6A4_subset, invalid_smiles_list
):
    X_smiles, Y = SLC6A4_subset.SMILES[:100], SLC6A4_subset.pXC50[:100]
    X_smiles = X_smiles.to_frame()

    # Ensure safe inference mode is off (default behavior)
    set_safe_inference_mode(smiles_pipeline, False)

    # Train the model
    smiles_pipeline.fit(X_smiles, Y)

    # Create a test set with invalid SMILES
    X_test = pd.DataFrame({"SMILES": X_smiles["SMILES"].tolist() + invalid_smiles_list})

    # Test prediction with invalid SMILES
    with pytest.raises(Exception):
        smiles_pipeline.predict(X_test)


@skip_pandas_output_test
def test_safeinference_wrapper_pandas_output(
    smiles_pipeline, SLC6A4_subset, pandas_output
):
    X_smiles = SLC6A4_subset.SMILES[:100].to_frame()

    # Set safe inference mode
    set_safe_inference_mode(smiles_pipeline, True)

    # Fit and transform (up to the FP step)
    result = smiles_pipeline[:-1].fit_transform(X_smiles)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == len(X_smiles)
    assert result.shape[1] == smiles_pipeline.named_steps["FP"].fpSize


@skip_pandas_output_test
def test_safeinference_wrapper_get_feature_names_out(smiles_pipeline):
    # Get feature names from the FP step
    feature_names = smiles_pipeline.named_steps["FP"].get_feature_names_out()
    assert len(feature_names) == smiles_pipeline.named_steps["FP"].fpSize
    assert all(isinstance(name, str) for name in feature_names)
