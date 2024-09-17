#For a non-scikit-learn check smiles sanitizer class

import pandas as pd
from rdkit import Chem


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
                print(f'Error in parsing {len(X_errors)} SMILES. Unparsable SMILES can be found in self.errors')

            self.errors = pd.DataFrame({'SMILES':X_errors, 'y':y_errors})

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
                print(f'Error in parsing {len(X_errors)} SMILES. Unparsable SMILES can be found in self.errors')

            self.errors = pd.DataFrame({'SMILES':X_errors})

            return X_out, X_errors
