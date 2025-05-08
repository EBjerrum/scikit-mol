# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Pipelining the scikit-mol transformer
#
# One of the very usable things with scikit-learn are their pipelines. With pipelines different scikit-learn transformers can be stacked and operated on just as a single model object. In this example we will build a simple model that can predict directly on RDKit molecules and then expand it to one that predicts directly on SMILES strings
#
# First some needed imports and a dataset

# %%
import os
import rdkit
from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import numpy as np

# %%
csv_file = "../tests/data/SLC6A4_active_excapedb_subset.csv"  # Hmm, maybe better to download directly
data = pd.read_csv(csv_file)
# %% [markdown]
# The dataset is a subset of the SLC6A4 actives from ExcapeDB. They are hand selected to give test set performance despite the small size, and are provided as example data only and should not be used to build serious QSAR models.
#
# We add RDKit mol objects to the dataframe with pandastools and check that all conversions went well.

# %%
PandasTools.AddMoleculeColumnToFrame(data, smilesCol="SMILES")
print(f"{data.ROMol.isna().sum()} out of {len(data)} SMILES failed in conversion")
# %% [markdown]
# Then, let's import some tools from scikit-learn and two transformers from scikit-mol

# %%
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from scikit_mol.fingerprints import MorganFingerprintTransformer
from scikit_mol.conversions import SmilesToMolTransformer

# %%
mol_list_train, mol_list_test, y_train, y_test = train_test_split(
    data.ROMol, data.pXC50, random_state=0
)

# %% [markdown]
# After a split into train and test, we'll build the first pipeline

# %%
pipe = Pipeline(
    [("mol_transformer", MorganFingerprintTransformer()), ("Regressor", Ridge())]
)
print(pipe)

# %% [markdown]
# We can do the fit by simply providing the list of RDKit molecule objects

# %%
pipe.fit(mol_list_train, y_train)
print(f"Train score is :{pipe.score(mol_list_train,y_train):0.2F}")
print(f"Test score is  :{pipe.score(mol_list_test, y_test):0.2F}")
# %% [markdown]
# Nevermind the performance, or the exact value of the prediction, this is for demonstration purpures. We can easily predict on lists of molecules

# %%
pipe.predict([Chem.MolFromSmiles("c1ccccc1C(=O)[OH]")])

# %% [markdown]
# We can also expand the already fitted pipeline, how about creating a pipeline that can predict directly from SMILES? With scikit-mol that is easy!

# %%
smiles_pipe = Pipeline(
    [("smiles_transformer", SmilesToMolTransformer()), ("pipe", pipe)]
)
print(smiles_pipe)

# %%
smiles_pipe.predict(["c1ccccc1C(=O)[OH]"])

# %% [markdown]
# From here, the pipelines could be pickled, and later loaded for easy prediction on RDKit molecule objects or SMILES in other scripts. The transformation with the MorganTransformer will be the same as during fitting, so no need to remember if radius 2 or 3 was used for this or that model, as it is already in the pipeline itself. If we need to see the parameters for a particular pipeline of model, we can always get the non default settings via print or all settings with .get_params().

# %%
smiles_pipe.get_params()
