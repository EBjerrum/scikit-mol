# checking that the new transformers can work within a scikitlearn pipeline of the kind
# Pipeline([("s2m", SmilesToMol()), ("FP", FPTransformer()), ("RF", RandomForestRegressor())])
# using some test data stored in ./data/SLC6A4_active_excape_subset.csv

# to run as
# pytest tests/test_transformers.py --> tests/test_transformers.py::test_transformer PASSED


import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.fingerprints import MACCSKeysFingerprintTransformer, RDKitFingerprintTransformer, AtomPairFingerprintTransformer, \
                                    TopologicalTorsionFingerprintTransformer, MorganFingerprintTransformer, SECFingerprintTransformer, \
                                    MHFingerprintTransformer, AvalonFingerprintTransformer


from fixtures import SLC6A4_subset

def test_transformer(SLC6A4_subset):
    # load some toy data for quick testing on a small number of samples
    X_smiles, Y = SLC6A4_subset.SMILES, SLC6A4_subset.pXC50
    X_train, X_test = X_smiles[:128], X_smiles[128:]
    Y_train, Y_test = Y[:128], Y[128:]

    # run FP with default parameters except when useCounts can be given as an argument
    FP_dict = {"MACCSTransformer": [MACCSKeysFingerprintTransformer, None],
               "RDKitFPTransformer": [RDKitFingerprintTransformer, None],
               "AtomPairFingerprintTransformer": [AtomPairFingerprintTransformer, False],
               "AtomPairFingerprintTransformer useCounts": [AtomPairFingerprintTransformer, True],
               "TopologicalTorsionFingerprintTransformer": [TopologicalTorsionFingerprintTransformer, False],
               "TopologicalTorsionFingerprintTransformer useCounts": [TopologicalTorsionFingerprintTransformer, True],
               "MorganTransformer": [MorganFingerprintTransformer, False],
               "MorganTransformer useCounts": [MorganFingerprintTransformer, True],
               "SECFingerprintTransformer": [SECFingerprintTransformer, None],
               "MHFingerprintTransformer": [MHFingerprintTransformer, None],
               'AvalonFingerprintTransformer': [AvalonFingerprintTransformer, None]}

    # fit on toy data and print train/test score if successful or collect the failed FP
    failed_FP = []
    for FP_name, (FP, useCounts) in FP_dict.items():
        try:
            print(f"\nrunning pipeline fitting and scoring for {FP_name} with useCounts={useCounts}")
            if useCounts is None:
                pipeline = Pipeline([("s2m", SmilesToMolTransformer()), ("FP", FP()), ("RF", RandomForestRegressor())])
            else:
                pipeline = Pipeline([("s2m", SmilesToMolTransformer()), ("FP", FP(useCounts=useCounts)), ("RF", RandomForestRegressor())])
            pipeline.fit(X_train, Y_train)
            train_score = pipeline.score(X_train, Y_train)
            test_score = pipeline.score(X_test, Y_test)
            print(f"\nfitting and scoring completed train_score={train_score}, test_score={test_score}")
        except:
            print(f"\n!!!! FAILED pipeline fitting and scoring for {FP_name} with useCounts={useCounts}")
            failed_FP.append(FP_name)
            pass

    # overall result
    assert len(failed_FP) == 0, f"the following FP have failed {failed_FP}"








