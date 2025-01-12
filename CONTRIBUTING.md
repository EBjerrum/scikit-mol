# Contribution

Thanks for your interest in contributing to the project. Please read on in the sections that apply.

## Slack channel

We have a slack channel for communication, ask for an invite: esbenbjerrum+scikit_mol@gmail.com
It's not really active and Slack wan't to be paid now. Maybe we can use Discord instead as slack is now deleting old threads.

## Installation

Clone and install in editable more like this

    git clone git@github.com:EBjerrum/scikit-mol.git
    pip install -e .[dev]

If you get issues that the editable mode install needs a setup.py, you should update your pip

## Code Quality

We use [ruff](https://github.com/astral-sh/ruff) to lint and format the code. The configuration is in the [ruff.toml](ruff.toml) file. The CI will fail if the code is not formatted correctly. You can run the linter and formatter locally with:

```sh
ruff format scikit_mol
ruff check --fix scikit_mol
```

We also have pre-commit hooks that will run the linter and formatter before you commit and we highly recommend you to use them. You can install them with:

```sh
pre-commit install
```
For more information on pre-commit see [documentation](https://pre-commit.com/).


## Adding transformers

The projects transformers subclasses the BaseEstimator and Transformer mixin classes from sklearn. Their documentation page contains information on what requisites are necessary [https://scikit-learn.org/stable/developers/develop.html](https://scikit-learn.org/stable/developers/develop.html). Most notably:

- The arguments accepted by **init** should all be keyword arguments with a default value.
- Every keyword argument accepted by **init** should correspond to an attribute on the instance.
- - There should be no logic, not even input validation, and the parameters should not be changed inside the **init** function.
    Scikit-learn classes depends on this in order to for e.g. the .get_params(), .set_params(), cloning abilities and representation rendering to work.
- With the new error handling, falsy objects need to return masked arrays or arrays with np.nan (for float dtype)

### Tips

- We have observed that some external tools used "exotic" types such at np.int64 when doing hyperparameter tuning. It is thus necessary do defensive programming to cast parameters to standard types before making calls to rdkit functions. This behaviour is tested in the test_parameter_types test

- @property getters and setters can be used if additional logic are needed when setting the attributes from the keywords while at the same time adhering to the sklearn requisites.

- Some RDKit features uses objects as generators which may not be picklable. If instantiated and added to the object as an attribute rather than instantiated at each function call for individual molecules, these should thus be removed and recreated via overloading the \_get_state() and \_set_state() methods. See MHFingerprintTransformer for an example.

## Module organisation

Currently we have multiple classes in the same file, if they are the same type. This may change in the future.

## Docstrings

We should ultimately consolidate on the NumPy docstring format [https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) which is also used by SciPy and other scikits.

## Typehints

parameters and output of methods should preferably be using typehints

## Testing

New transformer classes should be added to the pytest tests in the tests directory. A lot of tests are made general, and tests aspects of the transformers that are needed for sklearn compliance or other features. The transformer is then added to a fixture and can be added to the lists of transformer objects that are run by these test. Specific tests may also be necessary to set up. As exampe the assert_transformer_set_params needs a list of non-default parameters in order to set the set_params functionality of the object.
Scikit-Learn has a check_estimator that we should strive to get to work, some classes of scikit-mol currently does not pass all tests.

## Notebooks

Another way of contributing is by providing notebooks with examples on how to use the project to build models together with Scikit-Learn and other tools. There is both .ipynb and #%% delimited .py files in the notebook directory as the first are useful for online rendering on github, whereas the later is useful for sub version control.

There are three scripts for handling the notebooks and their associated python:percent scripts (with much nicer diff for git).

`pair_notebook.sh` create a pair of a notebook or percent:py script

`sync_notebooks.sh` uses jupytext to sync .py and ipynb. Jupytext is available via conda-forge or pip

`update_notebooks.sh` will sync, run and save the notebooks, expects a ipython kernel with scikit-mol installed called Python3.

## Release


_PyPi_

To relesase a new version on PyPi, you need to create and push new tag in v0.0.0 format then workflow will automatically build and upload the package to PyPi. Additonally the release draft with autogenerated notes and signed distribution files will be added to the GitHub release page. What is left is to publish the release, after checking that the notes are correct.


_Conda_
When you make a release on pypi the conda-forge bot will automatically make a PR that updates the Conda feedstock to the new version. If new dependencies or pins are changed on dependencies, those changes will need to be added to the PR. If there is just a pure code change then all we have do to is merge in the PR and that will update the package on conda-forge. See https://conda-forge.org/docs/maintainer/updating_pkgs/ for more information
