#%%
%load_ext autoreload 
%autoreload 2
from rdkit import Chem
from scikit_mol.transformers import MorganTransformer, SmilesToMol


#
smiles_list = ['c1ccccc1'] * 10
smilestomol = SmilesToMol()
mols = smilestomol.fit_transform(smiles_list)
mols[0]

#%%
X= [Chem.MolFromSmiles('c1ccccc1')]*10
t = MorganTransformer(useCounts=True)
X_t = t.fit_transform(X)
print(X_t.sum())
len(X_t)
# %% We can get the parameters
print(t.get_params())

# %% We can set the parameters (needed for scikit compatibility)
# It takes the parameters from __init__
t.set_params(nBits=1024, radius=3)
print(t.get_params(deep=True))


# %% Must be clonable
from sklearn import clone
t2 = clone(t)
t2.get_params()
# %%
