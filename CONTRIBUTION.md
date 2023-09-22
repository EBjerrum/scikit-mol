# Contribution

Thanks for your interest in contributing to the project. Please read on in the sections that apply.


## Installation
Clone and install in editable more like this

    git clone git@github.com:EBjerrum/scikit-mol.git
    pip install -e .

## Adding transformers
The projects transformers subclasses the BaseEstimator and Transformer mixin classes from sklearn. Their documentation page contains information on what requisites are necessary [https://scikit-learn.org/stable/developers/develop.html](https://scikit-learn.org/stable/developers/develop.html). Most notably:
* The arguments accepted by __init__ should all be keyword arguments with a default value.
* Every keyword argument accepted by __init__ should correspond to an attribute on the instance. 
* * There should be no logic, not even input validation, and the parameters should not be changed.
Scikit-learn classes depends on this in order to for e.g. the .get_params(), .set_params(), cloning abilities and representation rendering to work.

### Docstrings
We should ultimately consolidate on the numpy docstring format [https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) which is also used by SciPy and other scikits.

### Testing
New transformer classes should be added to the pytest tests in the tests directory. There may be a need for specific tests for the specific transformer, but it should also be added to the general tests that test sklearn necessary aspects of the transformer such as clonability. 

### Tips
* We have observed that some external tools used "exotic" types such at np.int64 when doing hyperparameter tuning. It is thus necessary to cast to standard types before making calls to rdkit functions. This behaviour is tested in the test_parameter_types test

* @property getters and setters can be used if additional logic are needed when setting the attributes from the keywords while at the same time adhering to the sklearn requisites. 

* Some RDKit features uses objects as generators which may not be picklable. If instantiated and added to the object rather than instantiated at each function call for individual molecules, these should thus be removed and recreated via overloading the _get_state() and _set_state() methods. See MHFingerprintTransformer for an example.



