# Scikit-Mol notebooks with examples

## Documentation

This is a collection of notebooks in the notebooks directory which demonstrates some different aspects and use cases

* [Basic Usage and fingerprint transformers](https://github.com/EBjerrum/scikit-mol/tree/main/notebooks/01_basic_usage.ipynb)
* [Descriptor transformer](https://github.com/EBjerrum/scikit-mol/tree/main/notebooks/02_descriptor_transformer.ipynb)
* [Pipelining with Scikit-Learn classes](https://github.com/EBjerrum/scikit-mol/tree/main/notebooks/03_example_pipeline.ipynb)
* [Molecular standardization](https://github.com/EBjerrum/scikit-mol/tree/main/notebooks/04_standardizer.ipynb)
* [Sanitizing SMILES input](https://github.com/EBjerrum/scikit-mol/tree/main/notebooks/05_smiles_sanitaztion.ipynb)
* [Integrated hyperparameter tuning of Scikit-Learn estimator and Scikit-Mol transformer](https://github.com/EBjerrum/scikit-mol/tree/main/notebooks/06_hyperparameter_tuning.ipynb)
* [Using parallel execution to speed up descriptor and fingerprint calculations](https://github.com/EBjerrum/scikit-mol/tree/main/notebooks/07_parallel_transforms.ipynb)


## Developers
There are three scripts for handling the notebooks and their associated python:percent scripts (with much nicer diff for git)

`pair_notebook.sh` create a pair of a notebook or percent:py script

`sync_notebooks.sh` uses jupytext to sync .py and ipynb. Jupytext is available via conda-forge or pip

`update_notebooks.sh` will sync, run and save the notebooks, expects a ipython kernel with scikit-mol installed called Python3. 
