#%%
import os
import rdkit
from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import numpy as np

#%%
!wget -c https://ndownloader.figshare.com/files/25747817 

#%%
data = pd.read_csv('25747817')
#%%
PandasTools.AddMoleculeColumnToFrame(data, smilesCol="SMILES")
# %% Any None mols?
data.ROMol.isna().sum()

# %%
len(data)
# %%
from scikit_mol.transformers import MorganTransformer
# %%
t = MorganTransformer()
# %%
t.fit_transform(data.ROMol)
# %%
_ = plt.hist(data.pXC50, bins=100)
#%%
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from scikit_mol.standardizer import Standardizer


#%%
mol_list_train, mol_list_test, y_train, y_test = train_test_split(data.ROMol, data.pXC50, random_state=0)

#%%
pipe = Pipeline([('mol_transformer', MorganTransformer()), ('Regressor', Ridge())])

# %%
pipe.fit(mol_list_train, y_train)
pipe.score(mol_list_train,y_train)
#%%
pipe.score(mol_list_test, y_test)

#%%
pipe.predict([Chem.MolFromSmiles('c1ccccc1C(=O)[OH]')])

#%% Now hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV

from sklearn.utils.fixes import loguniform

#%% Which keys do we have?

pipe.get_params().keys()

#%%

param_dist = {'Regressor__alpha': loguniform(1e-2, 1e3),
            "mol_transformer__nBits": [256,512,1024,2048,4096],
            'mol_transformer__radius':[1,2,3,4],
            'mol_transformer__useCounts': [True,False],
            'mol_transformer__useFeatures':[True,False]}

# %% From https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
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


# %% Appears standardizing is repeated for each run in the randomized search CV, irrespective of the caching argument.
# Probably the recommended way would be to prestandardize the data if there's no changes to the transformer, 
# and then add the standardizer in the inference pipeline.

standardizer = Standardizer()
mol_list_std_train = standardizer.transform(mol_list_train)


#%%
n_iter_search = 25
random_search = RandomizedSearchCV(
    pipe, param_distributions=param_dist, n_iter=n_iter_search
)
t0 = time()
random_search.fit(mol_list_std_train, y_train.values)
t1 = time()

print(f'Runtime: {t1-t0} for {n_iter_search} iterations)')

#%%
report(random_search.cv_results_)
# %% Building an inference pipeline, it appears our test-data was pretty standard
inference_pipe = Pipeline([('Standardizer', standardizer), ('best_estimator', random_search.best_estimator_)])

print(f'No Standardization {random_search.best_estimator_.score(mol_list_test, y_test)}')
print(f'With Standardization {inference_pipe.score(mol_list_test, y_test)}')
# %%
# Intergrating the Standardizer and challenge it with some different forms and salts of benzoic acid
smiles_list = ['c1ccccc1C(=O)[OH]', 'c1ccccc1C(=O)[O-]', 'c1ccccc1C(=O)[O-].[Na+]', 'c1ccccc1C(=O)[O][Na]', 'c1ccccc1C(=O)[O-].C[N+](C)C']
mols_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

print(f'Predictions with no standardization: {random_search.best_estimator_.predict(mols_list)}')
print(f'Predictions with standardization:    {inference_pipe.predict(mols_list)}')
