# %% [markdown]
# # Parallel calculations of transforms
#
# Scikit-mol has some support for parallel calculations of the fingerprints. It can be controlled via the parallel parameter or attribute. 
# Due to the overhead of the process with splitting the dataset, pickling it and sending it to child processes, it is not worthwhile for all combinations of dataset sizes and fingerprint types. 
# The faster the fingerprint, the larger the dataset needs to be for it to be worthwhile.

# first some imports of the usual suspects: RDKit, pandas, matplotlib, numpy and sklearn. New kid on the block is scikit-mol

# %%
import os
import rdkit
from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import numpy as np
# from sklearn.pipeline import Pipeline, make_pipeline
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import train_test_split
from scikit_mol.transformers import MorganTransformer, SmilesToMol
from scikit_mol.descriptors import Desc2DTransformer

# %% [markdown]
# We will need some data. There is a dataset with the SLC6A4 active compounds from ExcapeDB on Zenodo. The scikit-mol project uses a subset of this for testing, and the samples there has been specially selected to give good results in testing (it should therefore be used for any production modelling). If full_set is false, the fast subset will be used, and otherwise the full dataset will be downloaded if needed.

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


# %% First a demonstration of the speedup that can be had for the descriptor transformer
dataset_size = 500
parallel=False
start_method = 'fork'
transformer = Desc2DTransformer(parallel=parallel, start_method=start_method)

t0 = time.time()
X = transformer.transform(data.ROMol.iloc[0:dataset_size])
t = time.time()-t0
print(f"Calculation time on dataset of size {dataset_size}: {t:0.2F} seconds")

# %%
dataset_size = 500
parallel = True
start_method = 'fork'
transformer = Desc2DTransformer(parallel=parallel,  start_method=start_method)

t0 = time.time()
X = transformer.transform(data.ROMol.iloc[0:dataset_size])
t = time.time()-t0
print(f"Parallel calculation time on dataset of size {dataset_size}: {t:0.2F} seconds")
# %%

from itertools import product
import math


dataset_sizes = [10,25,100,250,500,1000, 2500, 5000, len(data)]
n_processes = [0] + [int(2**i) for i in range(1, int(math.log(8)/math.log(2))+1)]
Transformer = Desc2DTransformer
Transformer = MorganTransformer
results = pd.DataFrame()

for dataset_size, n_proc in product(dataset_sizes, n_processes):
    transformer = Transformer(parallel=n_proc)
    t0 = time.time()
    X = transformer.transform(data.ROMol.iloc[0:dataset_size])
    t = time.time()-t0
    print(f"{dataset_size} {n_proc} {t}")
    results.loc[n_proc, dataset_size, ] = t
    


display(results)


import seaborn as sns

from matplotlib import pyplot as plt
sns.heatmap(results.loc[0]/results, annot=True, cmap = "PiYG",center=1)
plt.title(f"Descriptor calculation parallelization speedup\n{Transformer.__name__}\n(SLC6A4 actives dataset)")
plt.xlabel("Dataset size")
plt.ylabel("Number of processes")


# %%