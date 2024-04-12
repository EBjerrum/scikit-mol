# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: rdkit2
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Parallel calculations of transforms
#
# Scikit-mol supports parallel calculations of fingerprints and descriptors. This feature can be activated and configured using the `parallel` parameter or the `.parallel` attribute after object instantiation.
#
# To begin, let's import the necessary libraries: RDKit and pandas. And of course, we'll also need to import scikit-mol, which is the new kid on the block.

# %%
from rdkit.Chem import PandasTools
import pandas as pd
import time


from scikit_mol.descriptors import MolecularDescriptorTransformer
from scikit_mol.fingerprints import MorganFingerprintTransformer
from scikit_mol.conversions import SmilesToMolTransformer

# %% [markdown]
# ## Obtaining the Data
#
# We'll need some data to work with, so we'll use a dataset of SLC6A4 active compounds from ExcapeDB that is available on Zenodo. Scikit-mol uses a subset of this dataset for testing purposes, and the samples have been specially selected to provide good results in testing. Note: This dataset should never be used for production modeling.
#
# In the code below, you can set full_set to True to download the full dataset. Otherwise, the smaller dataset will be used.

# %%
full_set = False

if full_set:
    csv_file = "SLC6A4_active_excape_export.csv"
    if not os.path.exists(csv_file):
        import urllib.request
        url = "https://ndownloader.figshare.com/files/25747817"
        urllib.request.urlretrieve(url, csv_file)
else:
    csv_file = '../tests/data/SLC6A4_active_excapedb_subset.csv'

# %%

data = pd.read_csv(csv_file)

PandasTools.AddMoleculeColumnToFrame(data, smilesCol="SMILES")
print(f"{data.ROMol.isna().sum()} out of {len(data)} SMILES failed in conversion")

# %% [markdown]
# ## Evaluating the Impact of Parallelism on Transformations
#
# Let's start by creating a baseline for our calculations without using parallelism.

# %% A demonstration of the speedup that can be had for the descriptor transformer
transformer = MolecularDescriptorTransformer(parallel=False)


# %%
def test_transformer(transformer):
    t0 = time.time()
    X = transformer.transform(data.ROMol)
    t = time.time()-t0
    print(f"Calculation time on dataset of size {len(data)}  with parallel={transformer.parallel}:\t{t:0.2F} seconds")
test_transformer(transformer)

# %% [markdown]
#
# Let's see if parallelism can help us speed up our transformations.

# %%

transformer = MolecularDescriptorTransformer(parallel=True)
test_transformer(transformer)

# %% [markdown]
# We've seen that parallelism can help speed up our transformations, with the degree of speedup depending on the number of CPU cores available. However, it's worth noting that there may be some overhead associated with the process of splitting the dataset, pickling objects and functions, and passing them to the parallel child processes. As a result, it may not always be worthwhile to use parallelism, particularly for smaller datasets or certain types of fingerprints.
#
# It's also worth noting that there are different methods for creating the child processes, with the default method on Linux being 'fork', while on Mac and Windows it's 'spawn'. The code we're using has been tested on Linux using the 'fork' method.
#
# Now, let's see how parallelism impacts another type of transformer.

# %% Some of the benchmarking plots
transformer = MorganFingerprintTransformer(parallel=False)
test_transformer(transformer)
transformer.parallel = True
test_transformer(transformer)


# %% [markdown]
# Interestingly, we observed that parallelism actually took longer to calculate the fingerprints in some cases, which is a perfect illustration of the overhead issue associated with parallelism. Generally, the faster the fingerprint calculation in itself, the larger the dataset needs to be for parallelism to be worthwhile. For example, the Descriptor transformer, which is one of the slowest, can benefit even for smaller datasets, while for faster fingerprint types like Morgan, Atompairs, and Topological Torsion fingerprints, the dataset needs to be larger.
#
# ![ Relation ship between throughput and parallel speedup](https://github.com/EBjerrum/scikit-mol/raw/main/notebooks/images/max_speedup_vs_throughput.png "Not all fingerprints are equally fast and benefit the same from parallelism")
#
# We've also included a series of plots below, showing the speedup over serial for different numbers of cores used for different dataset sizes. These timings were taken on a 16 core machine (32 Hyperthreads). Only the largest datasets (>10,000 samples) would make it worthwhile to parallelize Morgan, Atompairs, and Topological Torsions. SECfingerprint, MACCS keys, and RDKitFP are intermediate and would benefit from parallelism when the dataset size is larger, say >500. Descriptors, on the other hand, benefit almost immediately even for the smallest datasets (>100 samples).
#
# ![Atompairs fingerprint](https://github.com/EBjerrum/scikit-mol/raw/main/notebooks/images/AtomPairFingerprintTransformer_par.png "Atompairs fingerprint speedup")
#
# ![Descriptors calculation speedup](https://github.com/EBjerrum/scikit-mol/raw/main/notebooks/images/Desc2DTransformer_par.png "Descriptors calculation speedup")
#
# ![MACCS keys speedup](https://github.com/EBjerrum/scikit-mol/raw/main/notebooks/images/MACCSTransformer_par.png "MACCS keys speedup")
#
# ![Morgan fingerprint speedup](https://github.com/EBjerrum/scikit-mol/raw/main/notebooks/images/MorganTransformer_par.png "Morgan fingerprint speedup")
#
# ![RDKit fingerprint speedup](https://github.com/EBjerrum/scikit-mol/raw/main/notebooks/images/RDKitFPTransformer_par.png "RDKit fingerprint speedup")
#
# ![SEC fingerprint speedup](https://github.com/EBjerrum/scikit-mol/raw/main/notebooks/images/SECFingerprintTransformer_par.png "SEC fingerprint speedup")
#
# ![TopologicalTorsion fingerprint speedup](https://github.com/EBjerrum/scikit-mol/raw/main/notebooks/images/TopologicalTorsionFingerprintTransformer_par.png "TopologicalTorsion fingerprint speedup")
#
#
#
#
