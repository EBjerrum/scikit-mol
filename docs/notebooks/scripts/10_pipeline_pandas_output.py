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
# # Preserving feature information in DataFrames
#
# This notebook highlights the ability of scikit-mol transformers to return data in DataFrames with meaningful column names. Some use-cases of this feature are illustrated.
#
# ***NOTE***: The goal of this notebook is to highlight the advantages of storing transformer output in DataFrames with meaningful column names. This notebook should *not* be considered a set of good practices for training and evaluating QSAR pipelines. The performance metrics of the resulting pipelines are pretty bad: the dataset they have been trained on is pretty small. Tuning the hyperparameters of the Random Forest regressor model (maximum depth of the trees, maximum features to consider when splitting...) can be beneficial. Also including dimensionality reduction / feature selection techniques can be beneficial, since pipelines use a high number of features for a small number of samples. Of course, to further reduce the risk of overfitting, the best hyperparameters and preprocessing techniques should be chosen in cross validation.

# %%
from pathlib import Path
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import FunctionTransformer
from scikit_mol.conversions import SmilesToMolTransformer
from sklearn.compose import make_column_selector, make_column_transformer
from scikit_mol.standardizer import Standardizer
from scikit_mol.descriptors import MolecularDescriptorTransformer
from scikit_mol.fingerprints import MorganFingerprintTransformer

# %%
csv_file = Path("../tests/data/SLC6A4_active_excapedb_subset.csv")
assert csv_file.is_file()
data = pd.read_csv(csv_file)
data.drop_duplicates(subset="Ambit_InchiKey", inplace=True)

# %% [markdown]
# Let's split the dataset in training and test, so we will be able to use the test set to evaluate the performance of models trained on the training set.

# %%
data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)

# %%
column_smiles = "SMILES"
column_target = "pXC50"

smis_train = data_train.loc[:, column_smiles]
target_train = data_train.loc[:, column_target]
smis_test = data_test.loc[:, column_smiles]
target_test = data_test.loc[:, column_target]

# %% [markdown]
# ## Descriptors pipeline that returns DataFrames
#
# Define a pipeline that:
#
# - converts SMILES strings to Mol objects
# - standardizes the molecules
# - computes molecular descriptors
#
# Then, we will configure the pipeline to return output in Pandas DataFrames.
# The column names will correspond to the descriptor names.

# %%
descriptors_pipeline = make_pipeline(
    SmilesToMolTransformer(),
    Standardizer(),
    MolecularDescriptorTransformer(),
)
descriptors_pipeline.set_output(transform="pandas")

# %%
df_descriptors = descriptors_pipeline.transform(smis_train)
df_descriptors

# %% [markdown]
# All scikit-mol transformers are now compatible with the scikit-learn [set_output API](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html).
#
# Let's define a pipeline that returns Morgan fingerprints in a DataFrame.
# Columns will be named with the pattern `fp_morgan_1`, `fp_morgan_2`, ...,`fp_morgan_N`.

# %%
fingerprints_pipeline = make_pipeline(
    SmilesToMolTransformer(),
    Standardizer(),
    MorganFingerprintTransformer(),
)
fingerprints_pipeline.set_output(transform="pandas")

# %%
df_fingerprints = fingerprints_pipeline.transform(smis_train)
df_fingerprints

# %% [markdown]
# ## Analyze feature importance of regression pipeline
#
# Making the transformation steps return Pandas DataFrames instead of NumPy arrays makes it easy to analyze the feature importance of regression models.
#
# Let's define a pipeline that, starting from SMILES strings, computes descriptors and uses them to predict the target with a Random Forest (RF) regression model. Since descriptors values have very different ranges, it's better to scale them before passing them to the RF regression model.

# %%
params_random_forest = {
    "max_depth": 5,  # Setting a low maximum depth to avoid overfitting
}

regression_pipeline = make_pipeline(
    SmilesToMolTransformer(),
    Standardizer(),
    MolecularDescriptorTransformer(),
    StandardScaler(),  # Scale the descriptors
    RandomForestRegressor(**params_random_forest),
)
regression_pipeline.set_output(transform="pandas")

# %%
regression_pipeline.fit(smis_train, target_train)
pred_test = regression_pipeline.predict(smis_test)


# %% [markdown]
# Let's define a simple function to compute regression metrics, and use it to evaluate the test set performance of the pipeline.


# %%
def compute_metrics(y_true, y_pred):
    result = {
        "RMSE": mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False),
        "MAE": mean_absolute_error(y_true=y_true, y_pred=y_pred),
        "R2": r2_score(y_true=y_true, y_pred=y_pred),
    }
    return result


performance = compute_metrics(y_true=target_test, y_pred=pred_test)
performance

# %%
regressor = regression_pipeline[-1]
regressor

# %% [markdown]
# Since we used `set_output(transform="pandas")` on the pipeline, the last step of the pipeline (the regression model) has the descriptor names in the `feature_names_in_` attribute. We can use them and the `feature_importances_` attribute to easily analyze the feature importances.

# %%
df_importance = pd.DataFrame(
    {
        "feature": regressor.feature_names_in_,
        "importance": regressor.feature_importances_,
    }
)
df_importance

# %% [markdown]
# Sort the features by most to least important:

# %%
df_importance.sort_values(
    by="importance", ascending=False, inplace=True, ignore_index=True
)
df_importance

# %%
n_top_features = 5
top_features = df_importance.head(n_top_features).loc[:, "feature"].tolist()
print(f"The {n_top_features} most important features are:")
for feature in top_features:
    print(feature)

# %% [markdown]
# ## Including external features
#
# The ability to keep the results of scikit-learn transformers in DataFrames with meaningful column names simplifies the task of analyzing the resulting models.
#
# Another good use-case is when we want to combine cheminformatics features from some other tool (QM packages, Deep Learning embeddings...) with the traditional cheminformatics features available in scikit-mol. It will be easier to keep track of the features that come from scikit-mol and the features that come from other tools, if they are stored in DataFrames with meaningful column names.
#
# Let's include features from the popular [CDDD](https://github.com/jrwnter/cddd) tool. CDDD is a Variational AutoEncoder Deep Learning model, and the CDDD features are the inner latent space representations of the SMILES. For additional details, have a look at the original CDDD paper:
#
# > R. Winter, F. Montanari, F. Noé, and D.-A. Clevert, “Learning continuous and data-driven molecular descriptors by translating equivalent chemical representations,” Chem. Sci., vol. 10, no. 6, pp. 1692–1701, Feb. 2019, [doi: 10.1039/C8SC04175J](https://doi.org/10.1039/C8SC04175J).
#
# We have precomputed these features and stored them in a file:

# %%
file_cddd_features = Path("../tests/data/CDDD_SLC6A4_active_excapedb_subset.csv.gz")
assert file_cddd_features.is_file()
df_cddd = pd.read_csv(file_cddd_features)
df_cddd


# %% [markdown]
# The CDDD features are stored in columns `cddd_1`, `cddd_2`, ..., `cddd_512`. The file has the identifier column `Ambit_InchiKey` that we can use to combine the CDDD features with the rest of the data:


# %%
def combine_datasets(data, cddd):
    data_combined = pd.merge(
        left=data,
        right=cddd,
        on="Ambit_InchiKey",
        how="inner",
        validate="one_to_one",
    )
    return data_combined


data_combined_train = combine_datasets(data_train, df_cddd)
data_combined_test = combine_datasets(data_test, df_cddd)

# %%
# The CDDD descriptors couldn't be computed for few molecules and can be removed as commented out below. The Datafile is now prefiltered
# target_train = data_train.loc[data_train["Ambit_InchiKey"].isin(data_combined_train["Ambit_InchiKey"]), column_target]
# target_test = data_test.loc[data_test["Ambit_InchiKey"].isin(data_combined_test["Ambit_InchiKey"]), column_target]

target_train = data_combined_train.loc[:, column_target]
target_test = data_combined_test.loc[:, column_target]

# %% [markdown]
# Now we can define a pipeline that uses the original SMILES column to compute the descriptors available in scikit-mol, then concatenates them with the pre-computed CDDD features, and uses all of them to train the regression model. We will need a slightly more complex pipeline with column selectors and transformers. For more details on this technique, please refer to the [official documentation](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html).
#
# Since we will keep everything in DataFrames, it will be easier to understand the effect of the CDDD features and the traditional descriptors available in scikit-mol.

# %%
# A pipeline to compute scikit-mol descriptors
descriptors_pipeline = make_pipeline(
    SmilesToMolTransformer(),
    Standardizer(),
    MolecularDescriptorTransformer(),
)
# A pipeline that just passes the input data.
# We will use it to preserve the CDDD features and pass them to downstream steps.
identity_pipeline = make_pipeline(
    FunctionTransformer(),
)
combined_transformer = make_column_transformer(
    (descriptors_pipeline, make_column_selector(pattern="SMILES")),
    (identity_pipeline, make_column_selector(pattern=r"^cddd_\d+$")),
    remainder="drop",
)
combined_transformer

# %%
pipeline_combined = make_pipeline(
    combined_transformer,
    StandardScaler(),
    RandomForestRegressor(**params_random_forest),
)
pipeline_combined.set_output(transform="pandas")

# %%
pipeline_combined.fit(data_combined_train, target_train)
pred_combined_test = pipeline_combined.predict(data_combined_test)
performance_combined = compute_metrics(y_true=target_test, y_pred=pred_combined_test)
performance_combined

# %% [markdown]
# Let's combine the performance metrics obtained using only the scikit-mol descriptors as input features, and the performance metrics obtained using also the CDDD features:

# %%
df_performance = pd.DataFrame(
    [performance, performance_combined], index=["descriptors", "combined"]
)
df_performance

# %% [markdown]
# All performance metrics were improved by the inclusion of the CDDD features.
# Let's analyze the feature importances of the model:

# %%
regressor = pipeline_combined[-1]
df_importance = pd.DataFrame(
    {
        "feature": regressor.feature_names_in_,
        "importance": regressor.feature_importances_,
    }
).sort_values(by="importance", ascending=False, ignore_index=True)
df_importance

# %%
top_features = df_importance.head(n_top_features).loc[:, "feature"].tolist()
print(f"The {n_top_features} most important features are:")
for feature in top_features:
    print(feature)

# %% [markdown]
# As we can see, some CDDD features are among the most important features for the regression model.
#
# Note that since the pipeline is a combination of two pipelines, the column names were prefixed by `pipeline-1` (the scikit-mol descriptors pipeline) and `pipeline-2` (the pipeline that selects and preserves pre-computed CDDD features).
