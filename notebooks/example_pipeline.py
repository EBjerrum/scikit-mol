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

#%%
X_train, X_test, y_train, y_test = train_test_split(data.ROMol, data.pXC50, random_state=0)

# %%
pipe = Pipeline([('mol_transformer', MorganTransformer()), ('Regressor', Ridge())])
# %%
pipe.fit(X_train, y_train)
pipe.score(X_train,y_train)
#%%
pipe.score(X_test, y_test)

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

#%%
n_iter_search = 50
random_search = RandomizedSearchCV(
    pipe, param_distributions=param_dist, n_iter=n_iter_search
)

start = time()
random_search.fit(X_train.values, y_train.values)
print(
    "RandomizedSearchCV took %.2f seconds for %d candidates parameter settings."
    % ((time() - start), n_iter_search)
)
#%%
report(random_search.cv_results_)
# %%
random_search.best_estimator_.score(X_test, y_test)
# %%
