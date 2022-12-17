# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3.9.4 ('rdkit')
#     language: python
#     name: python3
# ---

# %%
import os
import rdkit
from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
#from scikit_mol.transformers import MorganTransformer, SmilesToMol
from scikit_mol.descriptors import Desc2DTransformer
from multiprocessing import Pool
import time


# %%
full_set = True

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

# %%
import time
dataset_size = 500

parallel=False
transformer = Desc2DTransformer(parallel=parallel)

t0 = time.time()
X = transformer.transform(data.ROMol.iloc[0:dataset_size])
t = time.time()-t0
print(t)

# %%
parallel = True
transformer = Desc2DTransformer(parallel=parallel)

t0 = time.time()
X = transformer.transform(data.ROMol.iloc[0:dataset_size])
t = time.time()-t0
print(t)

# %%
parallel = 2 #Takes slightly longer than True
transformer = Desc2DTransformer(parallel=parallel)

t0 = time.time()
X = transformer.transform(data.ROMol.iloc[0:dataset_size])
t = time.time()-t0
print(t)

# %%
from itertools import product


dataset_sizes = [10,100,250,500,1000, 2500, 5000]
n_processes = [0, 2, 3, 4]


results = pd.DataFrame()

for dataset_size, n_proc in product(dataset_sizes, n_processes):
    transformer = Desc2DTransformer(parallel=n_proc)
    t0 = time.time()
    X = transformer.transform(data.ROMol.iloc[0:dataset_size])
    t = time.time()-t0
    print(f"{dataset_size} {n_proc} {t}")
    results.loc[n_proc, dataset_size, ] = t
    


# %%
results

# %%
import seaborn as sns

# %%
from matplotlib import pyplot as plt
sns.heatmap(results.loc[0]/results, annot=True, cmap = "PiYG",center=1)
plt.title("Descriptor calculation parallelization speedup\n(SLC6A4 actives dataset)")
plt.xlabel("Dataset size")
plt.ylabel("Number of processes")
