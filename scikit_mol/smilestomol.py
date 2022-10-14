from rdkit import Chem
from sklearn.base import BaseEstimator, TransformerMixin


class SmilesToMol(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        #Nothing to do here
        return self

    def transform(self, X_smiles_list):
        # Unfortunately, transform is only X to X in Scikit-learn, so can't filter at this level
        # TODO: Return same type as put in (e.g. List to list, numpy to numpy, pandas Series to pandas series)
        X_out = []

        for smiles in X_smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                X_out.append(mol)
            else:
                raise ValueError(f'Issue with parsing SMILES {smiles}\nYou probably should use the scikit-mol.sanitizer.Sanitizer first')

        return X_out

        