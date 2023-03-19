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
# # Molecule standardization
# When building machine learning models of molecules, it is important to standardize the molecules. We often don't want different predictions just because things are drawn in slightly different forms, such as protonated or deprotanted carboxylic acids. Scikit-mol provides a very basic standardize transformer based on the molvs implementation in RDKit

# %%
from rdkit import Chem
from scikit_mol.standardizer import Standardizer
from scikit_mol.fingerprints import MorganTransformer
from scikit_mol.conversions import SmilesToMol
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

# %% [markdown]
# For demonstration let's create some molecules with different protonation states. The two first molecules are Benzoic acid and Sodium benzoate.

# %%
smiles_strings = ('c1ccccc1C(=O)[OH]','c1ccccc1C(=O)[O-].[Na+]','CC[NH+](C)C','CC[N+](C)(C)C',
       '[O-]CC(C(=O)[O-])C[NH+](C)C','[O-]CC(C(=O)[O-])C[N+](C)(C)C')

smi2mol = SmilesToMol()

mols  = smi2mol.transform(smiles_strings)
for mol in mols[0:2]:
    display(mol)

# %% [markdown]
# We can simply use the transformer directly and get a list of standardized molecules

# %%
# You can just run straight up like this. Note that neutralising is optional
standardizer = Standardizer(neutralize=True)
standard_mols = standardizer.transform(mols)
standard_smiles = smi2mol.inverse_transform(standard_mols)
standard_smiles

# %% [markdown]
# Some of the molecules were desalted and neutralized.
#
# A typical usecase would be to add the standardizer to a pipeline for prediction

# %%
# Typical use case is to use it in an sklearn pipeline, like below 
predictor = Ridge()

std_pipe = make_pipeline(SmilesToMol(), Standardizer(), MorganTransformer(useCounts=True), predictor)
nonstd_pipe = make_pipeline(SmilesToMol(), MorganTransformer(useCounts=True), predictor)

fake_y = range(len(smiles_strings))

std_pipe.fit(smiles_strings, fake_y)


print(f'Predictions with no standardization: {std_pipe.predict(smiles_strings)}')
print(f'Predictions with standardization:    {nonstd_pipe.predict(smiles_strings)}')


# %% [markdown]
# As we can see, the predictions with the standardizer and without are different. The two first molecules were benzoic acid and sodium benzoate, which with the standardized pipeline is predicted as the same, but differently with the nonstandardized pipeline. Wheter we want to make the prediction on the parent compound, or predict the exact form, will of course depend on the use-case, but now there is at least a way to handle it easily in pipelined predictors.
#
# The example also demonstrate another feature. We created the ridge regressor before creating the two pipelines. Fitting one of the pipelines thus also updated the object in the other pipeline. This can be useful for building inference pipelines that takes in SMILES molecules, but rather do the fitting on already converted and standardized molecules. However, be aware that the crossvalidation classes of scikit-learn may clone the estimators internally when doing the search loop, which would break this interdependence, and necessitate the rebuilding of the inference pipeline.
#
# If we had fitted the non standardizing pipeline, the model would have been different as shown below, as some of the molecules would be perceived different by the Ridge regressor.

# %%
nonstd_pipe.fit(smiles_strings, fake_y)
print(f'Predictions with no standardization: {std_pipe.predict(smiles_strings)}')
print(f'Predictions with standardization:    {nonstd_pipe.predict(smiles_strings)}')
