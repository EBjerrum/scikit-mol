# %% [markdown]
# Demo of the descriptor module

# %%
from rdkit import Chem
import numpy as np
import matplotlib.pyplot as plt
from scikit_mol.descriptors import Desc2DTransformer
# %% [markdown]
# What descriptors are available 

descriptor = Desc2DTransformer()
available_descriptors = descriptor.available_descriptors
print(f"There are {len(available_descriptors)} available descriptors")

# %% [markdown]
# We can transform molecules to their descriptors

# %%
smiles_list = ["CCCC", "c1ccccc1"]
mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

features = descriptor.transform(mols)
plt.plot(np.array(features).T)

# %% [markdown]
# If we only want some of them, we can specify that at descriptor creation time

# %%
some_descriptors = Desc2DTransformer(desc_list=['HeavyAtomCount', 'FractionCSP3', 'RingCount', 'MolLogP', 'MolWt'])
print(f"Selected descriptors are {some_descriptors.selected_descriptors}")
features = some_descriptors.transform(mols)
plt.plot(np.array(features).T)

# %%
print(some_descriptors)
# %% [markdown]
# There is an error check of the specified descriptor list

# %%
some_faulty_descriptors = Desc2DTransformer(desc_list=['Color', 'Icecream content', 'ChokolateDarkness', 'Content42', 'MolWt'])
# %%
