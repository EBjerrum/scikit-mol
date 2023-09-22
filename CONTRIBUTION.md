# Contribution

Thanks for your interest in contributing to the project. Please read on in the sections that apply.


## Slack channel
We have a slack channel for communication, ask for an invite: esbenbjerrum+scikit_mol@gmail.com


## Installation
Clone and install in editable more like this

    git clone git@github.com:EBjerrum/scikit-mol.git
    pip install -e .[dev]

If you get issues that the editable mode install needs a setup.py, you should update your pip

## Adding transformers
The projects transformers subclasses the BaseEstimator and Transformer mixin classes from sklearn. Their documentation page contains information on what requisites are necessary [https://scikit-learn.org/stable/developers/develop.html](https://scikit-learn.org/stable/developers/develop.html). Most notably:
* The arguments accepted by __init__ should all be keyword arguments with a default value.
* Every keyword argument accepted by __init__ should correspond to an attribute on the instance. 
* * There should be no logic, not even input validation, and the parameters should not be changed.
Scikit-learn classes depends on this in order to for e.g. the .get_params(), .set_params(), cloning abilities and representation rendering to work.

### Tips
* We have observed that some external tools used "exotic" types such at np.int64 when doing hyperparameter tuning. It is thus necessary to cast to standard types before making calls to rdkit functions. This behaviour is tested in the test_parameter_types test

* @property getters and setters can be used if additional logic are needed when setting the attributes from the keywords while at the same time adhering to the sklearn requisites. 

* Some RDKit features uses objects as generators which may not be picklable. If instantiated and added to the object as an attribute rather than instantiated at each function call for individual molecules, these should thus be removed and recreated via overloading the _get_state() and _set_state() methods. See MHFingerprintTransformer for an example.


## Module organisation
Currently we have multiple classes in the same file, if they are the same type. This may change in the future.

## Docstrings
We should ultimately consolidate on the NumPy docstring format [https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) which is also used by SciPy and other scikits.

## Typehints
parameters and output of methods should preferably be using typehints

## Testing
New transformer classes should be added to the pytest tests in the tests directory. A lot of tests are made general, and tests aspects of the transformers that are needed for sklearn compliance or other features. The transformer is then added to a fixture and can be added to the lists of transformer objects that are run by these test. Specific tests may also be necessary to set up. As exampe the assert_transformer_set_params needs a list of non-default parameters in order to set the set_params functionality of the object.

## Notebooks
Another way of contributing is by providing notebooks with examples on how to use the project to build models together with Scikit-Learn and other tools. There is both .ipynb and #%% delimited .py files in the notebook directory as the first are useful for online rendering on github, whereas the later is useful for sub version control.

There are three scripts for handling the notebooks and their associated python:percent scripts (with much nicer diff for git).

`pair_notebook.sh` create a pair of a notebook or percent:py script

`sync_notebooks.sh` uses jupytext to sync .py and ipynb. Jupytext is available via conda-forge or pip

`update_notebooks.sh` will sync, run and save the notebooks, expects a ipython kernel with scikit-mol installed called Python3. 





