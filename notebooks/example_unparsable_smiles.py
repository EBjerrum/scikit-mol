# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3.9.9 ('rdkit')
#     language: python
#     name: python3
# ---

# %% [markdown]
# Often we are faced with datasets which has SMILES that rdkit doesn't want to sanitize. This can be human entry errors, or differences between RDKits more strict sanitazion and other toolkits implementations of the parser. e.g. RDKit will not handle a tetravalent nitrogen when it has no charge, where other toolkits may simply build the graph anyway, disregarding the issues with the valence rules or guessing that the nitrogen should have a charge.

# %%
import pandas as pd
from rdkit.Chem import PandasTools


# !wget -c https://ndownloader.figshare.com/files/25747817 

data = pd.read_csv('25747817')
PandasTools.AddMoleculeColumnToFrame(data, smilesCol="SMILES")

print(f'Dataset contains {data.ROMol.isna().sum()} unparsable mols')

# %% [markdown]
# Our toy dataset didn't contain any issues, but lets simulate that we had it, and show what we can do about it

# %%
smiles_list = list(data.SMILES.values)
y_values = list(data.pXC50.values)
smiles_list.append('CN(C)(C)(C)')
y_values.append(1000)

# %% [markdown]
# If we use these SMILES for the scikit-learn pipeline, we would face an error, so we need to check and clean the dataset first.

# %%
from scikit_mol.sanitizer import CheckSmilesSanitazion
smileschecker = CheckSmilesSanitazion()
smiles_list_valid, y_valid, X_errors, y_errors = smileschecker.sanitize(smiles_list, y_values)

# %% [markdown]
# Now the smiles_list_valid should be all valid and the y_values filtered as well. Errors are returned, but also accesible after the call to .sanitize() in the .errors property

# %%
smileschecker.errors

# %% [markdown]
# The checker can also be used only on X

# %%
smiles_list_valid, X_errors = smileschecker.sanitize(smiles_list)
smileschecker.errors

# %%
