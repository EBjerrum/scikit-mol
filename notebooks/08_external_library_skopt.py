#%% Needs scikit-optimize
#!mamba install scikit-optimize -c conda-forge


#%%
import numpy as np
import pandas as pd
# %%
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from scikit_mol.transformers import MorganTransformer, SmilesToMol

from sklearn.pipeline import make_pipeline

#%%
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from skopt import gp_minimize
# %%
data = pd.read_csv("/home/esben/git/scikit-mol/notebooks/SLC6A4_active_excape_export.csv")
data['ROMol'] = SmilesToMol().transform(data.SMILES)
 
#%%
pipe = make_pipeline(MorganTransformer(), Ridge())
pipe
# %%
print(pipe.get_params())

#%%
max_bits = 4096

morgan_space = [
    Categorical([True, False], name='morgantransformer__useCounts'),
    Categorical([True, False], name='morgantransformer__useFeatures'),
    Integer(512,max_bits, name='morgantransformer__nBits', dtype=int),
    Integer(1,3, name='morgantransformer__radius', dtype=int)
]


regressor_space = [Real(1e-2, 1e3, "log-uniform", name='ridge__alpha')]

search_space = morgan_space + regressor_space
# %%
@use_named_args(search_space)
def objective(**params):
    pipe.set_params(**params)

    return -np.mean(cross_val_score(pipe, data.ROMol, data.pXC50, cv=2, n_jobs=-1,
                                    scoring="neg_mean_absolute_error"))
# %% THIS takes forever on my machine with a GradientBoostingRegressor

pipe_gp = gp_minimize(objective, search_space, n_calls=10, random_state=0)

"Best score=%.4f" % pipe_gp.fun
# %%
print("""Best parameters:""")
print({param.name:value for param,value in zip(pipe_gp.space, pipe_gp.x) })
#%%
from skopt.plots import plot_convergence
plot_convergence(pipe_gp)
# %%
