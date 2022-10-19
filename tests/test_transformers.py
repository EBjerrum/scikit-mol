# checking that the new transformers can work within a scikitlearn pipeline of the kind
# Pipeline([("s2m", SmilesToMol()), ("FP", FPTransformer()), ("RF", RandomForestRegressor())])
# using some test data stored in ./data/SLC6A4_active_excape_export.csv


import pytest
from scikit_mol.transformers import MACCSTransformer, RDKitFPTransformer, AtomPairFingerprintTransformer, \
                                    TopologicalTorsionFingerprintTransformer, MorganTransformer

def test_transformer():
    try:
        print(f"\nrunning pipeline fitting and scoring for {FP_name} with useCounts={useCounts}")
        if useCounts is None:
            pipeline = Pipeline([("s2m", SmilesToMol()), ("FP", FP()), ("RF", RandomForestRegressor())])
        else:
            pipeline = Pipeline([("s2m", SmilesToMol()), ("FP", FP(useCounts=useCounts)), ("RF", RandomForestRegressor())])
        pipeline.fit(X_train, Y_train)
        train_score = pipeline.score(X_train, Y_train)
        test_score = pipeline.score(X_test, Y_test)
        print(f"\nfitting and scoring completed train_score={train_score}, test_score={test_score}")
    except:
        print(f"\n!!!! FAILED pipeline fitting and scoring for {FP_name} with useCounts={useCounts}")
        failed_FP.append(FP_name)
        pass


if __name__ == '__main__':
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from scikit_mol.transformers import SmilesToMol

    # load some toy data for quick testing on a small number of samples
    data = pd.read_csv("./data/SLC6A4_active_excape_export.csv")
    X_smiles, Y = data.SMILES, data.pXC50
    X_train, X_test = X_smiles[:128], X_smiles[128:256]
    Y_train, Y_test = Y[:128], Y[128:256]

    # run FP with default parameters except when useCounts can be given as an argument
    FP_dict = {"MACCSTransformer": [MACCSTransformer, None],
               "RDKitFPTransformer": [RDKitFPTransformer, None],
               "AtomPairFingerprintTransformer": [AtomPairFingerprintTransformer, False],
               "AtomPairFingerprintTransformer useCounts": [AtomPairFingerprintTransformer, True],
               "TopologicalTorsionFingerprintTransformer": [TopologicalTorsionFingerprintTransformer, False],
               "TopologicalTorsionFingerprintTransformer useCounts": [TopologicalTorsionFingerprintTransformer, True],
               "MorganTransformer": [MorganTransformer, False],
               "MorganTransformer useCounts": [MorganTransformer, True]}

    # fit on toy data and print train/test score if successful or collect the failed FP
    failed_FP = []
    for FP_name, (FP, useCounts) in FP_dict.items():
        test_transformer()

    # overall result
    assert len(failed_FP) == 0, f"the following FP have failed {failed_FP}"








