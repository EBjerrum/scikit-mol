#%%

import os
import pandas as pd
import seaborn as sns
import numpy as np
import glob

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from matplotlib.colors import FuncNorm
import math
from matplotlib.colors import TwoSlopeNorm

#%%
results_dir = 'parallel_results'

files = glob.glob(f"{results_dir}/Morgan*csv")

files


#%%
def make_plot(results, transformer_name):
    ax = sns.heatmap(results.loc[0]/results, annot=True, cmap = "PiYG", center=10, norm=FuncNorm([np.log, np.exp]), vmax=32)
    ax.set_title(f"Descriptor calculation parallelization speedup\n{transformer_name}\n(SLC6A4 actives dataset)")
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Number of processes")
    

#%%
for filepath in files:
    transformer_name, ext = os.path.splitext(os.path.basename(filepath))
    results = pd.read_csv(filepath, index_col=0)
    ax = make_plot(results, transformer_name)
    display(plt.show())



#%%
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
              "potato", "wheat", "barley"]
farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
           "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])

fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
im = ax.imshow(harvest, vmin=0, vmax=32)
im2 = ax2.imshow(harvest)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(farmers)), labels=farmers)
ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)


cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("cbarlabel", rotation=-90, va="bottom")

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, f"{np.log(harvest[i, j]):0.2F}", ha="center", va="center", color="w")

ax.set_title("Harvest of local farmers (in tons/year)")
fig.tight_layout()
plt.show()

#%%
def make_plot_2(results, transformer_name):

    x_labels = results.columns
    y_labels = results.index
    #times = results.loc[0]/results
    times = results/results.loc[0]

    divnorm = TwoSlopeNorm( vcenter=1,
                        #vmin=times.min().min(), 
                        vmin=1/32,
                        #vmax=times.max().max()
                        vmax=2)
    #divnorm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=32)

    fig, ax = plt.subplots()
    im = ax.imshow(times,  cmap = "PiYG_r", norm=divnorm)#, vmin=0, vmax=32)
    #fig2, ax2 = plt.subplots()
    #im2 = ax2.imshow(results)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)


    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Fraction of single-threaded performance per dataset", rotation=-90, va="bottom")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, f"{times.iloc[i, j]:0.2F}", ha="center", va="center", color="black")

    mol_pr_sec = [int(c) for c in results.columns]/results.iloc[0]

    ax.set_title(f"Descriptor calculation parallelization times\n{transformer_name}\nMaximum throughput single-threaded: {mol_pr_sec.max():0.0F} mol/s")
    fig.tight_layout()
    return fig

#%%
files = glob.glob(f"{results_dir}/*csv")

files

#%%
for filepath in files:
    transformer_name, ext = os.path.splitext(os.path.basename(filepath))
    results = pd.read_csv(filepath, index_col=0)
    fig = make_plot_2(results, transformer_name)
    fig.savefig(f"images/{transformer_name}_par.png")
    fig.savefig(f"images/{transformer_name}_par.svg")



#%%
max_speedup = []
mol_pr_secs = []
names = []

for filepath in files:
    transformer_name, ext = os.path.splitext(os.path.basename(filepath))
    results = pd.read_csv(filepath, index_col=0)
    mol_pr_sec = ([int(c) for c in results.columns]/results.iloc[0]).max()
    mol_pr_secs.append(mol_pr_sec)

    speedup = (results.loc[0]/results).max().max()
    max_speedup.append(speedup)

    names.append(transformer_name)

# %%
plt.scatter(mol_pr_secs, max_speedup)
#plt.yscale('log')
# %%
sort_mask = np.argsort(mol_pr_secs)

x = np.array(mol_pr_secs)[sort_mask]
y = np.array(max_speedup)[sort_mask]
labels = np.array(names)[sort_mask]

plt.plot(x, y, marker='o')

for x_i,y_i,label in zip(x,y,labels):


    if label == "MorganTransformer":
        y_p = y_i - 0.2
    else:
        y_p = y_i + 0.2
    
    if label in ['TopologicalTorsionFingerprintTransformer',
       'AtomPairFingerprintTransformer', 'MorganTransformer']:
        x_p = 300
    else:
        x_p = x_i
    
    plt.annotate(label, (x_p, y_p))

plt.ylabel('Maximum speedup observed')
plt.xlabel("Singlethreaded throughput (molecules/second)")
plt.title("Relationship between maximum speedup and transformer througput")
#plt.yscale('log')
plt.xscale('log')

plt.savefig("images/max_speedup_vs_throughput.png")
plt.savefig("images/max_speedup_vs_throughput.svg")
# %%
