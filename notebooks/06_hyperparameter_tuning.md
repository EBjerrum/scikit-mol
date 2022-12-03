# Full example: Hyperparameter tuning


```python
import os
import rdkit
from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import numpy as np

csv_file = "../tests/data/SLC6A4_active_excapedb_subset.csv" # Hmm, maybe better to download directly
data = pd.read_csv(csv_file)

PandasTools.AddMoleculeColumnToFrame(data, smilesCol="SMILES")
print(f"{data.ROMol.isna().sum()} out of {len(data)} SMILES failed in conversion")

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from scikit_mol.transformers import MorganTransformer, SmilesToMol

mol_list_train, mol_list_test, y_train, y_test = train_test_split(data.ROMol, data.pXC50, random_state=0)

pipe = Pipeline([('mol_transformer', MorganTransformer()), ('Regressor', Ridge())])
```

    0 out of 200 SMILES failed in conversion



```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform
```


```python

pipe.get_params().keys()
```




    dict_keys(['memory', 'steps', 'verbose', 'mol_transformer', 'Regressor', 'mol_transformer__nBits', 'mol_transformer__radius', 'mol_transformer__useBondTypes', 'mol_transformer__useChirality', 'mol_transformer__useCounts', 'mol_transformer__useFeatures', 'Regressor__alpha', 'Regressor__copy_X', 'Regressor__fit_intercept', 'Regressor__max_iter', 'Regressor__normalize', 'Regressor__positive', 'Regressor__random_state', 'Regressor__solver', 'Regressor__tol'])




```python

param_dist = {'Regressor__alpha': loguniform(1e-2, 1e3),
            "mol_transformer__nBits": [256,512,1024,2048,4096],
            'mol_transformer__radius':[1,2,3,4],
            'mol_transformer__useCounts': [True,False],
            'mol_transformer__useFeatures':[True,False]}
```


```python
#run randomized search

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")
```


```python
# Probably the recommended way would be to prestandardize the data if there's no changes to the transformer, 
# and then add the standardizer in the inference pipeline.

from scikit_mol.standardizer import Standardizer

standardizer = Standardizer()
mol_list_std_train = standardizer.transform(mol_list_train)
```


```python
n_iter_search = 25
random_search = RandomizedSearchCV(
    pipe, param_distributions=param_dist, n_iter=n_iter_search
)
t0 = time()
random_search.fit(mol_list_std_train, y_train.values)
t1 = time()

print(f'Runtime: {t1-t0} for {n_iter_search} iterations)')
```

    Runtime: 4.731405973434448 for 25 iterations)



```python
report(random_search.cv_results_)
```

    Model with rank: 1
    Mean validation score: 0.604 (std: 0.130)
    Parameters: {'Regressor__alpha': 2.3337181365429247, 'mol_transformer__nBits': 1024, 'mol_transformer__radius': 2, 'mol_transformer__useCounts': False, 'mol_transformer__useFeatures': False}
    
    Model with rank: 2
    Mean validation score: 0.566 (std: 0.121)
    Parameters: {'Regressor__alpha': 1.6333109391336837, 'mol_transformer__nBits': 4096, 'mol_transformer__radius': 3, 'mol_transformer__useCounts': False, 'mol_transformer__useFeatures': False}
    
    Model with rank: 3
    Mean validation score: 0.515 (std: 0.163)
    Parameters: {'Regressor__alpha': 0.02342781901545695, 'mol_transformer__nBits': 512, 'mol_transformer__radius': 2, 'mol_transformer__useCounts': False, 'mol_transformer__useFeatures': False}
    



```python
inference_pipe = Pipeline([('Standardizer', standardizer), ('best_estimator', random_search.best_estimator_)])

print(f'No Standardization {random_search.best_estimator_.score(mol_list_test, y_test)}')
print(f'With Standardization {inference_pipe.score(mol_list_test, y_test)}')
```

    No Standardization 0.5472809065513343
    With Standardization 0.5472809065513343



```python
# Intergrating the Standardizer and challenge it with some different forms and salts of benzoic acid
smiles_list = ['c1ccccc1C(=O)[OH]', 'c1ccccc1C(=O)[O-]', 'c1ccccc1C(=O)[O-].[Na+]', 'c1ccccc1C(=O)[O][Na]', 'c1ccccc1C(=O)[O-].C[N+](C)C']
mols_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

print(f'Predictions with no standardization: {random_search.best_estimator_.predict(mols_list)}')
print(f'Predictions with standardization:    {inference_pipe.predict(mols_list)}')
```

    Predictions with no standardization: [6.11738406 6.19188974 6.19188974 6.1574595  6.25186147]
    Predictions with standardization:    [6.11738406 6.11738406 6.11738406 6.11738406 6.11738406]



```python

```
