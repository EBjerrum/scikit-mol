# ---
# jupyter:
#   jupytext:
#     formats: docs//notebooks//ipynb,docs//notebooks//scripts//py:percent
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
# # Safe inference mode
#
# I think everyone which have worked with SMILES and RDKit sooner or later come across a SMILES that doesn't parse. It can happen if the SMILES was produced with a different toolkit that are less strict with e.g. valence rules, or maybe a characher was missing in the copying from the email. During curation of the dataset for training models, these SMILES need to be identfied and eventually fixed or removed. But what happens when we are finished with our modelling? What kind of molecules and SMILES will a user of the model send for the model in the future when it's in deployment. What kind of SMILES will a generative model create that we need to predict? We don't know and we won't know. So it's kind of crucial to be able to handle these situations. Scikit-Learn models usually simply explodes the entire batch that are being predicted. This is where safe_inference_mode was introduced in Scikit-Mol. With the introduction all transformers got a safe inference mode, where they handle invalid input. How they handle it depends a bit on the transformer, so we will go through the different usual steps and see how things have changed with the introduction of the safe inference mode.
#
# NOTE! In the following demonstration I switch on the safe inference mode individually for demonstration purposes. I would not recommend to do that while building and training models, instead I would switch it on _after_ training and evaluation (more on that later). Otherwise, there's a risk to train on the 2% of a dataset that didn't fail....
#
# First some imports and test SMILES and molecules.

# %%
from rdkit import Chem
from scikit_mol.conversions import SmilesToMolTransformer

# We have some deprecation warnings, we are adressing them, but they just distract from this demonstration
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

smiles = ["C1=CC=C(C=C1)F", "C1=CC=C(C=C1)O", "C1=CC=C(C=C1)N", "C1=CC=C(C=C1)Cl"]
smiles_with_invalid = smiles + ["N(C)(C)(C)C", "I'm not a SMILES"]

smi2mol = SmilesToMolTransformer(safe_inference_mode=True)

mols_with_invalid = smi2mol.transform(smiles_with_invalid)
mols_with_invalid

# %% [markdown]
# Without the safe inference mode, the transformation would simply fail, but now we get the expected array back with our RDKit molecules and a last entry which is an object of the type InvalidMol. InvalidMol is simply a placeholder that tells what step failed the conversion and the error. InvalidMol evaluates to `False` in boolean contexts, so it gets easy to filter away and handle in `if`s and list comprehensions. As example:

# %%
[mol for mol in mols_with_invalid if mol]

# %% [markdown]
# or

# %%
mask = mols_with_invalid.astype(bool)
mols_with_invalid[mask]

# %% [markdown]
# Having a failsafe SmilesToMol conversion leads us to next step, featurization. The transformers in safe inference mode now return a NumPy masked array instead of a regular NumPy array. It simply evaluates the incoming mols in a boolean context, so e.g. `None`, `np.nan` and other Python objects that evaluates to False will also get masked (i.e. if you use a dataframe with an ROMol column produced with the PandasTools utility)

# %%
from scikit_mol.fingerprints import MorganFingerprintTransformer

mfp = MorganFingerprintTransformer(radius=2, fpSize=25, safe_inference_mode=True)
fps = mfp.transform(mols_with_invalid)
fps


# %% [markdown]
# However, currently scikit-learn models accepts masked arrays, but they do not respect the mask! So if you fed it directly to the model to train, it would seemingly work, but the invalid samples would all have the fill_value, meaning you could get weird results. Instead, we need the last part of the puzzle, the SafeInferenceWrapper class.

# %%
from scikit_mol.safeinference import SafeInferenceWrapper
from sklearn.linear_model import LogisticRegression
import numpy as np

regressor = LogisticRegression()
wrapper = SafeInferenceWrapper(regressor, safe_inference_mode=True)
wrapper.fit(fps, [0, 1, 0, 1, 0, 1])
wrapper.predict(fps)


# %% [markdown]
#

# %% [markdown]
# The prediction went fine both in fit and in prediction, where the result shows `nan` for the invalid entries. However, please note fit in sage_inference_mode is not recommended in a training session, but you are warned and not blocked, because maybe you know what you do and do it on purpose.
# The SafeInferenceMapper both handles rows that are masked in masked arrays, but also checks rows for nonfinite values and filters these away. Sometimes some descriptors may return an inf or nan, even though the molecule itself is valid. The masking of nonfinite values can be switched off, maybe you are using a model that can handle missing data and only want to filter away invalid molecules.
#
# ## Setting safe_inference_mode post-training
# As I said before I believe in catching errors and fixing those during training, but what do we do when we need to switch on safe inference mode for all objects in a pipeline? There's of course a tool for that, so lets demo that:

# %%
from scikit_mol.safeinference import set_safe_inference_mode
from sklearn.pipeline import Pipeline

pipe = Pipeline(
    [
        ("smi2mol", SmilesToMolTransformer()),
        ("mfp", MorganFingerprintTransformer(radius=2, fpSize=25)),
        ("safe_regressor", SafeInferenceWrapper(LogisticRegression())),
    ]
)

pipe.fit(smiles, [1, 0, 1, 0])

print("Without safe inference mode:")
try:
    pipe.predict(smiles_with_invalid)
except Exception as e:
    print("Prediction failed with exception: ", e)
print()

set_safe_inference_mode(pipe, True)

print("With safe inference mode:")
print(pipe.predict(smiles_with_invalid))

# %% [markdown]
# We see that the prediction fail without safe inference mode, and proceeds when it's conveniently set by the `set_safe_inference_mode` utility. The model is now ready for save and reuse in a more failsafe manner :-)

# %% [markdown]
# ## Combining safe_inference_mode with pandas output
# One potential issue can happen when we combine the safe_inference_mode with Pandas output mode of the transformers. It will work, but depending on the batch something surprising can happen due to the way that Pandas converts masked Numpy arrays. Let me demonstrate the issue, first we predict a batch without any errors.

# %%
mfp.set_output(transform="pandas")

mols = smi2mol.transform(smiles)

fps = mfp.transform(mols)
fps

# %% [markdown]
# Then lets see if we transform a batch with an invalid molecule:

# %%
fps = mfp.transform(mols_with_invalid)
fps

# %% [markdown]
# The second output is no longer integers, but floats. As most sklearn models cast input arrays to float32 internally, this difference is likely benign, but that's not guaranteed! Thus, if you want to use pandas output for your production models, do check that the final outputs are the same for the valid rows, with and without a single invalid row. Alternatively the dtype for the output of the transformer can be switched to float for consistency if it's supported by the transformer.

# %%
mfp_float = MorganFingerprintTransformer(
    radius=2, fpSize=25, safe_inference_mode=True, dtype=np.float32
)
mfp_float.set_output(transform="pandas")
fps = mfp_float.transform(mols)
fps

# %% [markdown]
# I hope this new feature of Scikit-Mol will make it even easier to handle models, even when used in environments without SMILES or molecule validity guarantees.
