# %%
from rdkit import Chem
from scikit_mol.standardizer import Standardizer
from scikit_mol.transformers import MorganTransformer
from sklearn.pipeline import make_pipeline

# %%
# Prep some RdKit molecules
smis = ('c1ccccc1C(=O)[O-]','c1ccccc1C(=O)[O-].[Na+]','CC[NH+](C)C','CC[N+](C)(C)C',
       '[O-]CC(C(=O)[O-])C[NH+](C)C','[O-]CC(C(=O)[O-])C[N+](C)(C)C')
X = [Chem.MolFromSmiles(smi) for smi in smis]

# %%
# You can just run straight up like this. Note that neutralising is optional
t = make_pipeline(Standardizer(neutralize=True))
X_t = t.fit_transform(X)
print(len(X_t))

# %%
# Typical use case is to use it in an sklearn pipeline, like below 
t2 = make_pipeline(Standardizer(), MorganTransformer(useCounts=True))
X_t2 = t2.fit_transform(X)
print(len(X_t2))
print(X_t2[0])