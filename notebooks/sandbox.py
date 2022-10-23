#%%
%load_ext autoreload 
%autoreload 2
from rdkit import Chem
import numpy as np
from scikit_mol.transformers import MorganTransformer, SmilesToMol


#%%
from scikit_mol.transformers import SmilesToMol
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
# %% Draft  pytest

tr = MorganTransformer()
mols_list = [Chem.MolFromSmiles(smi) for smi in ['Cc1ccccc1', 'c1cccnc1']]


def assert_transformer_set_params(tr_class, new_params):
    tr = tr_class()
    fps_default = tr.transform(mols_list)

    tr.set_params(**new_params)
    new_tr = MorganTransformer(**new_params)

    fps_reset_params = tr.transform(mols_list)
    fps_init_new_params = new_tr.transform(mols_list)
    # Now fp_default should not be the same as fp_reset_params,
    assert(~np.any([np.array_equal(fp_default, fp_reset_params) for fp_default, fp_reset_params in zip(fps_default, fps_reset_params)]))
    # fp_reset_params and fp_init_new_params should be the same
    assert(np.all([np.array_equal(fp_init_new_params, fp_reset_params) for fp_init_new_params, fp_reset_params in zip(fps_init_new_params, fps_reset_params)]))


new_params = {'nBits': 1024,
            'radius': 3,
            'useBondTypes': False,
            'useChirality': True,
            'useCounts': True,
            'useFeatures': True}



assert_transformer_set_params(MorganTransformer, new_params)
# %% But this wouldn't ca
from scikit_mol.transformers import MorganTransformer, MACCSTransformer, RDKitFPTransformer, AtomPairFingerprintTransformer, TopologicalTorsionFingerprintTransformer

# %%
MACCSTransformer().get_params()
AtomPairFingerprintTransformer().get_params()# %%

#%%
TopologicalTorsionFingerprintTransformer().get_params()

# %%
chiral_smiles_list = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in  [
                'N[C@@H](C)C(=O)O',
                'C1C[C@H]2CCCC[C@H]2CC1']]
# %%
mols = [Chem.MolFromSmiles(smi) for smi in chiral_smiles_list]
# %%
tr = TopologicalTorsionFingerprintTransformer(includeChirality=True)
# %%
tr.transform(mols)
# %%
