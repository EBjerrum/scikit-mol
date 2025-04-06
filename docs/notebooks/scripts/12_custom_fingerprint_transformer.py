# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Creating custom fingerprint transformers
#
# If the default fingerprint transformers provided by the scikit-mol library are not enough for your needs, you can create your own custom fingerprint transformers. In this notebook, we will show you how to do this.
#
# Note that base classes are partially stable and may change in the future versions of the library. We will try to keep the changes minimal and provide a migration guide if necessary. This notebook is also will be updated accordingly.

# %% [markdown]
# ## Basics
#
# For now we recommend you to use the `BaseFpsTransformer` class as a base class for your custom fingerprint transformers. This class provides a simple interface to create fingerprint transformers that can be used with the scikit-mol library.
#
# To create your custom fingerprint transformer, you need to create a class that inherits from the `BaseFpsTransformer` class and implement the `_transform_mol` method. This method should take a single RDKit molecule object as input and return a fingerprint as a numpy array.

# %%
from scikit_mol.fingerprints.baseclasses import BaseFpsTransformer
import numpy as np
from rdkit import Chem

class DummyFingerprintTransformer(BaseFpsTransformer):
    def __init__(self, fpSize=64, n_jobs=1, safe_inference_mode = False):
        self.fpSize = fpSize
        super().__init__(n_jobs=n_jobs, safe_inference_mode=safe_inference_mode, name="dummy")

    def _transform_mol(self, mol):
        return mol.GetNumAtoms() * np.ones(self.fpSize)
    
trans = DummyFingerprintTransformer(n_jobs=4)
mols = [Chem.MolFromSmiles('CC')] * 100
trans.transform(mols)

# %% [markdown]
# ## Non-pickable objects
# When working with some of the `rdkit` function and classes you will often discover that some of the are unpickable objects. This means that they cannot be serialized and deserialized using the `pickle` module. This is a problem when you want to use the parallelization (controlled by the `n_jobs` parameter).
#
# Any non-pickable object in the transformer attributes should be initialized in the `__init__` method of the transforme from the other *picklable* arguments.
#
# In the example below, we will create a custom fingerprint transformer that uses the Morgan fingerprint with radius **2** and **1024** bits. Used generator is unpickable, but it can be created during the initialization of the transformer from the picklable `maxPath` and `fpSize` arguments.

# %%
from rdkit.Chem import rdFingerprintGenerator

class UnpickableFingerprintTransformer(BaseFpsTransformer):
    def __init__(self, fpSize=1024, n_jobs=1, safe_inference_mode=False, **kwargs):
        self.fpSize = fpSize
        super().__init__(n_jobs=n_jobs, safe_inference_mode=safe_inference_mode, **kwargs)
        self.fp_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=2, fpSize=self.fpSize)

    def _transform_mol(self, mol):
        return self.fp_gen.GetFingerprintAsNumPy(mol)
    
trans = UnpickableFingerprintTransformer(n_jobs=4, fpSize=512)
trans.transform(mols)


# %% [markdown]
# Non-pickable object should not be present among the `__init__` arguments of the transformer. Doing so will prevent them to be pickled to recreate a transformer instance in the worker processes. If you for some reason need to pass a non-pickable object to the transformer you can do so (**highly discouraged**, please [open the issue](https://github.com/EBjerrum/scikit-mol/issues), maybe we will be able to help you do it better) by using the transformer in the sequential mode (i.e. `n_jobs=1`).

# %%
class BadTransformer(BaseFpsTransformer):
    def __init__(self, generator, n_jobs=1):
        self.generator = generator
        super().__init__(n_jobs=n_jobs)
    def _transform_mol(self, mol):
        return self.generator.GetFingerprint(mol)


fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=10)
BadTransformer(fp_gen, n_jobs=1).transform(mols)
print("n_jobs=1 is fine")
try:
    BadTransformer(fp_gen, n_jobs=2).transform(mols)
except Exception as e:
    print("n_jobs=2 is not fine, because the generator passed as an argument is not picklable")
    print(f"Error msg: {e}")


# %% [markdown]
# ## Fingerprint name
#
# To use the fingerptint in the `pandas` output mode it needes to know the name of the fingerprint and the number of bits (features) in it to generate the columns names. The number of feature is derived from `fpSize` attribute 
#
# And the name resolution works as follows (in order of priority):
# - if the fingerprint has a name set during the initialization of the base class, it is used
# - if the name of the class follows the pattern `XFingerprintTransformer`, the name (`fp_X`) is extracted from it
# - as a last resort, the name is set to name of the class

# %%
class NamedTansformer1(UnpickableFingerprintTransformer):
    pass

class NamedTansformer2(UnpickableFingerprintTransformer):
    def __init__(self):
        super().__init__(name="fp_fancy")

class FancyFingerprintTransformer(UnpickableFingerprintTransformer):
    pass

print(NamedTansformer1().get_feature_names_out())
print(NamedTansformer2().get_feature_names_out())
print(FancyFingerprintTransformer().get_feature_names_out())

