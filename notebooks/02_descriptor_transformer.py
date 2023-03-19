# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# # Desc2DTransformer: RDKit descriptors transformer
#
# The descriptors transformer can convert molecules into a list of RDKit descriptors. It largely follows the API of the other transformers, but has a few extra methods and properties to manage the descriptors.

# %%
from rdkit import Chem
import numpy as np
import matplotlib.pyplot as plt
from scikit_mol.descriptors import MolecularDescriptorTransformer
# %% [markdown]
# After instantiation of the descriptor transformer, we can query which descriptors it found available in the RDKit framework.

# %%
descriptor = MolecularDescriptorTransformer()
available_descriptors = descriptor.available_descriptors
print(f"There are {len(available_descriptors)} available descriptors")
print(f"The first five descriptor names: {available_descriptors[:5]}")

# %% [markdown]
# We can transform molecules to their descriptor profiles

# %%
smiles_list = ["CCCC", "c1ccccc1"]
mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

features = descriptor.transform(mols)
_ = plt.plot(np.array(features).T)

# %% [markdown]
# If we only want some of them, this can be specified at object instantiation.

# %%
some_descriptors = MolecularDescriptorTransformer(desc_list=['HeavyAtomCount', 'FractionCSP3', 'RingCount', 'MolLogP', 'MolWt'])
print(f"Selected descriptors are {some_descriptors.selected_descriptors}")
features = some_descriptors.transform(mols)

# %% [markdown]
# If we want to update the selected descriptors on an already existing object, this can be done via the .set_params() method

# %%
print(some_descriptors.set_params(desc_list=['HeavyAtomCount', 'FractionCSP3', 'RingCount']))

# %%
