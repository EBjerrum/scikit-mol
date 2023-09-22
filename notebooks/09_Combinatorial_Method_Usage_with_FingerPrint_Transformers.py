# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: aniEnv
#     language: python
#     name: anienv
# ---

# %% [markdown]
# # Example: Using Multiple Different Fingerprint Transformer
#
# In this notebook we will explore how to evaluate the performance of machine learning models depending on different fingerprint transformers (Featurization techniques). This is an example, that you easily could adapt for many different combinations of featurizers, optimizaiton and other modelling techniques.
#
# Following steps will happen:
# * Data Parsing
# * Pipeline Building
# * Training Phase
# * Analysis
#
# Authors: @VincentAlexanderScholz, @RiesBen 
#
# ## Imports:
# First we will import all the stuff that we will need for our work.
#

# %%
import os
import numpy as np
import pandas as pd
from time import time
from matplotlib import pyplot as plt

from rdkit.Chem import PandasTools

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from scikit_mol import fingerprints


# %% [markdown]
# ## Get Data:
# In this step we will check if the SLC6A4 data set is already present or needs to be downloaded.
#
#
# **WARNING:** The Dataset is a simple and very well selected

# %%
full_set = False

# if not present download example data
if full_set:
    csv_file = "SLC6A4_active_excape_export.csv"
    if not os.path.exists(csv_file):
        import urllib.request
        url = "https://ndownloader.figshare.com/files/25747817"
        urllib.request.urlretrieve(url, csv_file)
else:
    csv_file = '../tests/data/SLC6A4_active_excapedb_subset.csv'

#Parse Database
data = pd.read_csv(csv_file)

PandasTools.AddMoleculeColumnToFrame(data, smilesCol="SMILES")
print(f"{data.ROMol.isna().sum()} out of {len(data)} SMILES failed in conversion")

# %% [markdown]
# ## Build Pipeline:
# In this stage we will build the Pipeline consisting of the featurization part (finger print transformers) and the model part (Ridge Regression).
#
# Note that the featurization in this section is an hyperparameter, living in `param_grid`, and the `"fp_transformer"` string is just a placeholder, being replaced during pipeline execution. 
#
# This way we can define multiple different scenarios in `param_grid`, that allow us to rapidly explore different combinations of settings and methodologies.

# %%

regressor = Ridge()
optimization_pipe = Pipeline([("fp_transformer", "fp_transformer"), # this is a placeholder for different transformers
                              ("regressor", regressor)])

param_grid = [ # Here pass different Options and Approaches
    {
        "fp_transformer": [fingerprints.MorganFingerprintTransformer(),
                          fingerprints.AvalonFingerprintTransformer()],
        "fp_transformer__nBits": [2**x for x in range(8,13)],
    },
    {
       "fp_transformer": [fingerprints.RDKitFingerprintTransformer(),
                         fingerprints.AtomPairFingerprintTransformer(),
                         fingerprints.MACCSKeysFingerprintTransformer()], 
    },
]

global_options = {
    "regressor__alpha": np.linspace(0.1,1,5),
}

[params.update(global_options) for params in param_grid]

param_grid

# %% [markdown]
# ## Train Model
# In this section, the combinatorial approaches are trained.

# %%
# Split Data
mol_list_train, mol_list_test, y_train, y_test = train_test_split(data.ROMol, data.pXC50, random_state=0)

# Define Search Process
grid = GridSearchCV(optimization_pipe, n_jobs=1,
                    param_grid=param_grid)

# Train
t0 = time()
grid.fit(mol_list_train, y_train.values)
t1 = time()

print(f'Runtime: {t1-t0:0.2F}')

# %% [markdown]
# ## Analysis
#
# Now let's investigate our results from the training stage. Which one is the best finger print method for this data set? Which parameters are optimal?

# %%
df_training_stats = pd.DataFrame(grid.cv_results_)
df_training_stats

# %%
# Best Fingerprint Method / Performance
res_dict = {}
for i, row in df_training_stats.iterrows():
    fp_name = row['param_fp_transformer'] 
    if(fp_name in res_dict and row['mean_test_score'] > res_dict[fp_name]["mean_test_score"]):
        res_dict[fp_name] = row.to_dict()
    elif(not fp_name in res_dict):
        res_dict[fp_name] = row.to_dict()
        
df = pd.DataFrame(list(res_dict.values()))
df =df.sort_values(by="mean_test_score")

#plot test score vs. approach
plt.figure(figsize=[14,5])
plt.bar(range(len(df)), df.mean_test_score, yerr=df.std_test_score)
plt.xticks(range(len(df)), df.param_fp_transformer, rotation=90, fontsize=14)
plt.ylabel("mean score", fontsize=14)
plt.title("Best Model of Fingerprint Transformer Type", fontsize=18)
pass


# %%
# Best Fingerprint Method / Performance
from collections import defaultdict
res_dict = defaultdict(list)
for i, row in df_training_stats.iterrows():
    fp_name = row['param_fp_transformer'] 
    if("Morgan" in str(fp_name)):
        res_dict[fp_name].append(row)

for fp_type, rows in res_dict.items():
    df = pd.DataFrame(rows)
    df =df.sort_values(by="mean_test_score")

    #plot test score vs. approach
    xlabels = map(lambda x: "_".join(x), zip(df.param_fp_transformer__nBits.astype(str), df.param_regressor__alpha.astype(str)))

        
    plt.figure(figsize=[14,5])
    plt.bar(range(len(df)), df.mean_test_score, yerr=df.std_test_score)
    plt.xticks(range(len(df)), xlabels, rotation=90, fontsize=14)
    plt.ylabel("mean score", fontsize=14)
    plt.xlabel("bitsize_alpha", fontsize=14)

    plt.title("Fingerprint Transformer "+str(fp_type).split("(")[0]+" per Bitsize", fontsize=18)
    pass


# %%
#plot ALL test score vs. approach
df =df_training_stats.sort_values(by="mean_test_score")

plt.figure(figsize=[16,9])
plt.bar(range(len(df)), df.mean_test_score, yerr=df.std_test_score)
plt.ylabel("mean score", fontsize=14)
plt.xticks(range(len(df))[::5], df.param_fp_transformer[::5], rotation=90, fontsize=14)
plt.title("test score vs. approach", fontsize=18)
pass
