# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3.9.4 ('rdkit')
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Full example: Hyperparameter tuning
#
# first some imports of the usual suspects: RDKit, pandas, matplotlib, numpy and sklearn. New kid on the block is scikit-mol

# %%
import os
import rdkit
from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from scikit_mol.fingerprints import MorganFingerprintTransformer
from scikit_mol.conversions import SmilesToMolTransformer

# %% [markdown]
# We will need some data. There is a dataset with the SLC6A4 active compounds from ExcapeDB on Zenodo. The scikit-mol project uses a subset of this for testing, and the samples there has been specially selected to give good results in testing (it should therefore be used for any production modelling). If full_set is false, the fast subset will be used, and otherwise the full dataset will be downloaded if needed.

# %%
full_set = False

if full_set:
    csv_file = "SLC6A4_active_excape_export.csv"
    if not os.path.exists(csv_file):
        import urllib.request
        url = "https://ndownloader.figshare.com/files/25747817"
        urllib.request.urlretrieve(url, csv_file)
else:
    csv_file = '../tests/data/SLC6A4_active_excapedb_subset.csv'

# %% [markdown]
# The CSV data is loaded into a Pandas dataframe and the PandasTools utility from RDKit is used to add a column with RDKit molecules

# %%

data = pd.read_csv(csv_file)

PandasTools.AddMoleculeColumnToFrame(data, smilesCol="SMILES")
print(f"{data.ROMol.isna().sum()} out of {len(data)} SMILES failed in conversion")

# %% [markdown]
# We use the train_test_split to, well, split the dataframe's molecule columns and pXC50 column into lists for train and testing

# %%

mol_list_train, mol_list_test, y_train, y_test = train_test_split(data.ROMol, data.pXC50, random_state=42)


# %% [markdown]
# We will standardize the molecules before modelling. This is best done before the hyperparameter optimizatiion of the featurization with the scikit-mol transformer and regression modelling, as the standardization is otherwise done for every loop in the hyperparameter optimization, which will make it take longer time.

# %%
# Probably the recommended way would be to prestandardize the data if there's no changes to the transformer, 
# and then add the standardizer in the inference pipeline.

from scikit_mol.standardizer import Standardizer

standardizer = Standardizer()
mol_list_std_train = standardizer.transform(mol_list_train)

# %% [markdown]
# A simple pipeline with a MorganTransformer and a Ridge() regression for demonstration.

# %%

moltransformer = MorganFingerprintTransformer()
regressor = Ridge()

optimization_pipe = make_pipeline(moltransformer, regressor)



# %% [markdown]
# For hyperparameter optimization we import the RandomizedSearchCV class from Scikit-Learn. It will try different random combinations of settings and use internal cross-validation to find the best model. In the end, it will fit the best found parameters on the full set. We also import loguniform, to get a better sampling of some of the parameters.

# %% Now hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform

# %% [markdown]
# With the pipelines, getting the names of the parameters to tune is a bit more tricky, as they are concatenations of the name of the step and the parameter with double underscores in between. We can get the available parameters from the pipeline with the get_params() method, and select the parameters we want to change from there.

# %% Which keys do we have?

optimization_pipe.get_params().keys()

# %% [markdown]
# We will tune the regularization strength of the Ridge regressor, and try out different parameters for the Morgan fingerprint, namely the number of bits, the radius of the fingerprint, wheter to use counts or bits and features.

# %%

param_dist = {'ridge__alpha': loguniform(1e-2, 1e3),
            "morganfingerprinttransformer__nBits": [256,512,1024,2048,4096],
            'morganfingerprinttransformer__radius':[1,2,3,4],
            'morganfingerprinttransformer__useCounts': [True,False],
            'morganfingerprinttransformer__useFeatures':[True,False]}

# %% [markdown]
# The report function was taken from [this example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py) from the scikit learn documentation.

# %% From https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
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


# %% [markdown]
# We will do 25 tries of random parameter sets, and see what comes out as the best one. If you are using the small example dataset, this should take some second, but may take some minutes with the full set.

# %%
n_iter_search = 25
random_search = RandomizedSearchCV(
    optimization_pipe, param_distributions=param_dist, n_iter=n_iter_search, cv=3
)
t0 = time()
random_search.fit(mol_list_std_train, y_train.values)
t1 = time()

print(f'Runtime: {t1-t0:0.2F} for {n_iter_search} iterations)')

# %%
report(random_search.cv_results_)
# %% [markdown]
# It can be interesting to see what combinations of hyperparameters gave good results for the cross-validation. Usually the number of bits are in the high end and radius is 2 to 4. But this can vary a bit, as we do a small number of tries for this demo. More extended search with more iterations could maybe find even better and more consistent. solutions

# %% [markdown]
# Let's see if standardization had any influence on this dataset. We build an inference pipeline that includes the standardization object and the best estimator, and run the best estimator directly on the list of test molecules

# %%
inference_pipe = make_pipeline(standardizer, random_search.best_estimator_)

print(f'No Standardization {random_search.best_estimator_.score(mol_list_test, y_test):0.4F}')
print(f'With Standardization {inference_pipe.score(mol_list_test, y_test):0.4F}')

# %% Building an inference pipeline, it appears our test-data was pretty standard [markdown]
# We see that the dataset already appeared to be in forms that are similar to the ones coming from the standardization. 
#
# Interestingly the test-set performance often seem to be better than the CV performance during the hyperparameter search. This may be due to the model being refit at the end of the search to the whole training dataset, as the refit parameter on the randomized_search object by default is true. The final model is thus fitted on more data than the individual models during training.
#
# To demonstrate the effect of standartization we can see the difference if we challenge the predictor with different forms of benzoic acid and benzoates.
# %%
# Intergrating the Standardizer and challenge it with some different forms and salts of benzoic acid
smiles_list = ['c1ccccc1C(=O)[OH]', 'c1ccccc1C(=O)[O-]', 'c1ccccc1C(=O)[O-].[Na+]', 'c1ccccc1C(=O)[O][Na]', 'c1ccccc1C(=O)[O-].C[N+](C)C']
mols_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

print(f'Predictions with no standardization: {random_search.best_estimator_.predict(mols_list)}')
print(f'Predictions with standardization:    {inference_pipe.predict(mols_list)}')

# %% [markdown]
# Without standardization we get variation in the predictions, but with the standardization object in place, we get the same results. If you want a model that gives different predictions for the different forms, either the standardization need to be removed or the settings changed.
#
# From here it should be easy to save the model using pickle, so that it can be loaded and used in other python projects. The pipeline carries both the standardization, the featurization and the prediction in one, easy to reuse object. If you want the model to be able to predict directly from SMILES strings, check out the SmilesToMol class, which is also available in Scikit-Mol :-)
#

# %% [markdown]
#
