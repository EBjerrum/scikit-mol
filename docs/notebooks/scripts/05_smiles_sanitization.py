# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3.9.4 ('rdkit')
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SMILES sanitation
# Sometimes we are faced with datasets which has SMILES that rdkit doesn't want to sanitize. This can be human entry errors, or differences between RDKits more strict sanitazion and other toolkits implementations of the parser. e.g. RDKit will not handle a tetravalent nitrogen when it has no charge, where other toolkits may simply build the graph anyway, disregarding the issues with the valence rules or guessing that the nitrogen should have a charge, where it could also by accident instead have a methyl group too many.

# %%
import pandas as pd
from rdkit.Chem import PandasTools

csv_file = "../tests/data/SLC6A4_active_excapedb_subset.csv"  # Hmm, maybe better to download directly
data = pd.read_csv(csv_file)


# %% [markdown]
# Now, this example dataset contain all sanitizable SMILES, so for demonstration purposes, we will corrupt one of them

# %%
data.loc[1, "SMILES"] = "CN(C)(C)(C)"

# %%

PandasTools.AddMoleculeColumnToFrame(data, smilesCol="SMILES")
print(f"Dataset contains {data.ROMol.isna().sum()} unparsable mols")


# %% [markdown]
# If we use these SMILES for the scikit-learn pipeline, we would face an error, so we need to check and clean the dataset first. The CheckSmilesSanitation can help us with that.

# %%
from scikit_mol.utilities import CheckSmilesSanitization

smileschecker = CheckSmilesSanitization()

smiles_list_valid, y_valid, smiles_errors, y_errors = smileschecker.sanitize(
    list(data.SMILES), list(data.pXC50)
)

# %% [markdown]
# Now the smiles_list_valid should be all valid and the y_values filtered as well. Errors are returned, but also accessible after the call to .sanitize() in the .errors property

# %%
smileschecker.errors

# %% [markdown]
# The checker can also be used only on X

# %%
smiles_list_valid, X_errors = smileschecker.sanitize(list(data.SMILES))
smileschecker.errors
