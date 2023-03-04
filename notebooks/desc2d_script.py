import os
import rdkit
from rdkit import Chem
from rdkit.Chem import PandasTools

import pandas as pd
#import matplotlib.pyplot as plt
from time import time
# import numpy as np
# from sklearn.pipeline import Pipeline, make_pipeline
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import train_test_split
# from scikit_mol.transformers import MorganTransformer, RDKitFPTransformer, SmilesToMol, MACCSTransformer
from scikit_mol.descriptors import Desc2DTransformer
# from multiprocessing import Pool
import time

print(f"rdkit:{rdkit.__version__}")

if __name__ == "__main__":
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
    print(f"{data.ROMol.isna().sum()} out of {len(data)} SMILES failed in conversion", flush=True)

    #%%
    dataset_size = len(data)

    bigdata = list(data.ROMol)

    len(bigdata)


    # # %% Problem seem to be if three has been a run before!!!
    dataset_size = 100
    parallel=False
    start_method = 'fork'
    transformer = Desc2DTransformer(parallel=parallel, start_method=start_method)
    transformer.calculators.CalcDescriptors(Chem.MolFromSmiles('c1ccccc1O'))

    # t0 = time.time()
    smiles = list(data.ROMol.iloc[0:dataset_size].apply(Chem.MolToSmiles))
    print(smiles)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    X = transformer.transform(mols) #Something
    #X = transformer.transform([Chem.MolFromSmiles('c1ccccc1O')]*100)
    # t = time.time()-t0
    # print(t)
    # self = transformer
    # self.calculators = self._get_desc_calculator()

    # #Weird, if another transformer object has been used to transform some mols, the next one fails if it runs in parallel????
    # del transformer
    # import gc
    # gc.collect()
    


    #data.ROMol.iloc[0:dataset_size]
    # %%

    dataset_size = 100
    parallel = 12
    start_method = 'fork' #forkserver and spawn requires the code to be protected in __main__
    transformer = Desc2DTransformer(parallel=parallel,  start_method=start_method)

    print("Starting test")
    t0 = time.time()
    X = transformer.transform(data.ROMol.iloc[0:dataset_size])
    t = time.time()-t0
    print(t)

    transformer = Desc2DTransformer(parallel=parallel,  start_method=start_method)

    print("Starting test 2")
    t0 = time.time()
    X = transformer.transform(data.ROMol.iloc[0:dataset_size])
    t = time.time()-t0
    print(t)
