# %%
import gc
import os
import joblib
# import rdkit
# from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
#import matplotlib.pyplot as plt
import time
#import numpy as np
# from sklearn.pipeline import Pipeline, make_pipeline
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import train_test_split
from scikit_mol.transformers import MACCSTransformer, MorganTransformer, RDKitFPTransformer, SmilesToMol, AtomPairFingerprintTransformer, TopologicalTorsionFingerprintTransformer,SECFingerprintTransformer
from scikit_mol.descriptors import Desc2DTransformer

# %% [markdown]
# We will need some data. There is a dataset with the SLC6A4 active compounds from ExcapeDB on Zenodo. The scikit-mol project uses a subset of this for testing, and the samples there has been specially selected to give good results in testing (it should therefore be used for any production modelling). If full_set is false, the fast subset will be used, and otherwise the full dataset will be downloaded if needed.

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


from itertools import product
import math


import math

def generate_numbers(start_number, max_number):
    numbers = []
    for i in range(int(math.log10(start_number)), int(math.log10(max_number))+1):
        for j in [1, 2.5, 5]:
            n = int(j * (10 ** i))
            if n >= start_number and n <= max_number:
                numbers.append(n)
    return numbers

def test_parallel(TransformerClass, dataset, dataset_sizes=None, max_processes=None):
    if not max_processes: max_processes= joblib.cpu_count(only_physical_cores=True)
    if not dataset_sizes: dataset_sizes = [10,25,100,250,500,1000, 2500, 5000, len(dataset)] #TODO make some clever math
    n_processes = [0] + [int(2**i) for i in range(1, int(math.log(max_processes)/math.log(2))+1)]    

    results = pd.DataFrame()

    for dataset_size in dataset_sizes:
        subset_data = list(dataset.sample(dataset_size).ROMol)
        for n_proc in n_processes:
            #for dataset_size, n_proc in product(dataset_sizes, n_processes):
            transformer = TransformerClass(parallel=n_proc)
            t0 = time.time()
            X = transformer.transform(subset_data)
            t = time.time()-t0
            del transformer
            gc.collect()
            print(f"{dataset_size} {n_proc} {t}")
            results.loc[n_proc, dataset_size, ] = t

    return results


#%%
savedir = 'parallel_results'
os.makedirs(savedir, exist_ok=True)

transformerlist = [MorganTransformer,  AtomPairFingerprintTransformer, TopologicalTorsionFingerprintTransformer,
                    RDKitFPTransformer, SECFingerprintTransformer, MACCSTransformer,  RDKitFPTransformer, Desc2DTransformer ]

for Transformer in transformerlist:
    results = test_parallel(Transformer, data, max_processes=32)
    results.to_csv(f"{savedir}/{Transformer.__name__}.csv")




# %%
