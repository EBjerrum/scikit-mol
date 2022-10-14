# scikit-mol
scikit-learn classes for molecular vectorization using RDKit


TODO:
    Expand number of fingerprint classes and featurizers
        AtomPairs
        TopologicalTorsions
        RDKit
        Descriptors
        LINGOS
        ...

    Make dictionary based FP class
        No Hashing, .fit() learns the keys of the dataset

    Make a basic standardarizer transformer class

    Make a SMILES to Mol transformer class

Make Notebook with examples
    Standalone usage
    Inclusion in pipeline
        Can transformers be used in parallel (e.g. to use both FP features and Descriptors at the same time?)
    Hyperparameter optimization via native Scikit-Classes
    Hyperparameter optimization via external optimizer e.g. https://scikit-optimize.github.io/stable/


Make basic unit-tests


Installation
    git clone git@github.com:EBjerrum/scikit-mol.git
    pip install -e .



Contributers:
    Please add yourself here, we'll properly markdown it later
