# For a non-scikit-learn check smiles sanitizer class

import warnings

import pandas as pd
from rdkit import Chem
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline


class CheckSmilesSanitazion:
    def __init__(self, return_mol=False):
        self.return_mol = return_mol
        self.errors = pd.DataFrame()

    def sanitize(self, X_smiles_list, y=None):
        if y:
            y_out = []
            X_out = []
            y_errors = []
            X_errors = []

            for smiles, y_value in zip(X_smiles_list, y):
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    if self.return_mol:
                        X_out.append(mol)
                    else:
                        X_out.append(smiles)
                    y_out.append(y_value)
                else:
                    X_errors.append(smiles)
                    y_errors.append(y_value)

            if X_errors:
                print(
                    f"Error in parsing {len(X_errors)} SMILES. Unparsable SMILES can be found in self.errors"
                )

            self.errors = pd.DataFrame({"SMILES": X_errors, "y": y_errors})

            return X_out, y_out, X_errors, y_errors

        else:
            X_out = []
            X_errors = []

            for smiles in X_smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    if self.return_mol:
                        X_out.append(mol)
                    else:
                        X_out.append(smiles)
                else:
                    X_errors.append(smiles)

            if X_errors:
                print(
                    f"Error in parsing {len(X_errors)} SMILES. Unparsable SMILES can be found in self.errors"
                )

            self.errors = pd.DataFrame({"SMILES": X_errors})

            return X_out, X_errors


def set_safe_inference_mode(estimator, value):
    """
    Recursively set the safe_inference_mode parameter for all compatible estimators.

    :param estimator: A scikit-learn estimator, pipeline, or custom wrapper
    :param value: Boolean value to set for safe_inference_mode
    """

    def _set_safe_inference_mode_recursive(est, val):
        if hasattr(est, "safe_inference_mode"):
            est.safe_inference_mode = val

        # Handle Pipeline
        if isinstance(est, Pipeline):
            for _, step in est.steps:
                _set_safe_inference_mode_recursive(step, val)

        # Handle FeatureUnion
        elif isinstance(est, FeatureUnion):
            for _, transformer in est.transformer_list:
                _set_safe_inference_mode_recursive(transformer, val)

        # Handle ColumnTransformer
        elif isinstance(est, ColumnTransformer):
            for _, transformer, _ in est.transformers:
                _set_safe_inference_mode_recursive(transformer, val)

        # Handle SafeInferenceWrapper
        elif hasattr(est, "estimator") and isinstance(est.estimator, BaseEstimator):
            _set_safe_inference_mode_recursive(est.estimator, val)

        # Handle other estimators with get_params
        elif isinstance(est, BaseEstimator):
            params = est.get_params(deep=False)
            for _, param_value in params.items():
                if isinstance(param_value, BaseEstimator):
                    _set_safe_inference_mode_recursive(param_value, val)

    # Apply the recursive function
    _set_safe_inference_mode_recursive(estimator, value)

    # Final check
    params = estimator.get_params(deep=True)
    mismatched_params = [
        key.rstrip("__safe_inference_mode")
        for key, val in params.items()
        if key.endswith("__safe_inference_mode") and val != value
    ]

    if mismatched_params:
        warnings.warn(
            f"The following components have 'safe_inference_mode' set to a different value than requested: {mismatched_params}. "
            "This could be due to nested estimators that were not properly handled.",
            UserWarning,
        )

    return estimator
