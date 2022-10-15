#%%
%load_ext autoreload 
%autoreload 2
from rdkit import Chem
from scikit_mol.transformers import MorganTransformer, SmilesToMol


#%%
from scikit_mol.smilestomol import SmilesToMol
smiles_list = ['c1ccccc1'] * 10
smilestomol = SmilesToMol()
mols = smilestomol.fit_transform(smiles_list)
mols[0]


#%%
from scikit_mol.transformers import SmilesToMol
smiles_list = ['c1ccccc1'] * 10
y = list(range(10))
y.append(1000)
smiles_list.append('Invalid')

#%% This should raise Value-error
smilestomol = SmilesToMol()
mols = smilestomol.fit_transform(smiles_list)

#%%
from scikit_mol.sanitizer import CheckSmilesSanitazion
smiles_list = ['c1ccccc1'] * 10
y = list(range(10))
y.append(1000)
smiles_list.append('Invalid')

sanitizer = CheckSmilesSanitazion(return_mol=False)

smiles_list_valid, y_valid, X_errors, y_errors = sanitizer.sanitize(smiles_list, y)

print(sanitizer.errors)
print(X_errors)

mols = smilestomol.fit_transform(smiles_list_valid)
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
