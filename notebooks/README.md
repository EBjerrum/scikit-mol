# Scikit-Mol notebooks with examples

## Documentation

This is a collection of notebooks in the notebooks directory which demonstrates some different aspects and use cases

* [Basic Usage and fingerprint transformers](https://github.com/EBjerrum/scikit-mol/blob/documentation/notebooks/01_basic_usage.ipynb)
* [Descriptor transformer](https://github.com/EBjerrum/scikit-mol/blob/documentation/notebooks/02_descriptor_transformer.ipynb)
* [Pipelining with Scikit-Learn classes](https://github.com/EBjerrum/scikit-mol/blob/documentation/notebooks/03_example_pipeline.ipynb)
* [Molecular standardization](https://github.com/EBjerrum/scikit-mol/blob/documentation/notebooks/04_standardizer.ipynb)
* [Sanitizing SMILES input](https://github.com/EBjerrum/scikit-mol/blob/documentation/notebooks/05_smiles_sanitaztion.ipynb)
* [Integrated hyperparameter tuning of Scikit-Learn estimator and Scikit-Mol transformer](https://github.com/EBjerrum/scikit-mol/blob/documentation/notebooks/06_hyperparameter_tuning.ipynb)


## Developers
There are two scripts for handling the notebooks and their associated python:percent scripts (with much nicer diff for git)
sync_notebooks.sh uses jupytext to sync .py and ipynb. Jupytext is available via conda-forge or pip

update_notebooks.sh will sync, run and save the notebooks, expects a ipython kernel with scikit-mol installed called Python3. 