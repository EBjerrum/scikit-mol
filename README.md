# scikit-mol

Scikit-Learn classes for molecular vectorization using RDKit

The intended usage is to be able to add molecular vectorization directly into scikit-learn pipelines, so that the final model directly predict on RDKit molecules or SMILES strings

As example with the needed scikit-learn and -mol imports and RDKit mol objects in the mol_list_train and _test lists:

    pipe = Pipeline([('mol_transformer', MorganTransformer()), ('Regressor', Ridge())])
    pipe.fit(mol_list_train, y_train)
    pipe.score(mol_list_test, y_test)
    pipe.predict([Chem.MolFromSmiles('c1ccccc1C(=O)C')])

    >>> array([4.93858815])

The scikit-learn compatibility should also make it easier to include the fingerprinting step in hyperparameter tuning with scikit-learns utilities

The first draft for the project was created at the [RDKIT UGM 2022 hackathon](https://github.com/rdkit/UGM_2022) 2022-October-14


## Implemented
* Transformer Classes
    * SmilesToMol
    * Desc2DTransformer
    * MACCSTransformer
    * RDKitFPTransformer
    * AtomPairFingerprintTransformer
    * TopologicalTorsionFingerprintTransformer
    * MorganTransformer
<br>

* Utilities
    * CheckSmilesSanitazion

## Installation
Users can install latest tagged release from pip

    pip install scikit-mol

Bleeding edge

    pip install git+https://github.com:EBjerrum/scikit-mol.git

Developers 

    git clone git@github.com:EBjerrum/scikit-mol.git
    pip install -e .

## Documentation
None yet, but there are some # %% delimted examples in the notebooks directory that have some demonstrations

## BUGS
Probably still


## TODO
* Make standardizer less 'chatty'
* Unit test coverage of classes
* Make further example notebooks
    * Standalone usage (not in pipeline)
    * Advanced pipelining
    * Hyperparameter optimization via external optimizer e.g. https://scikit-optimize.github.io/stable/

## Ideas
* LINGOS transformer


## Contributers:
* Esben Bjerrum, esben@cheminformania.com
* Carmen Esposito https://github.com/cespos
* Son Ha, sonha@uni-mainz.de
* Oh-hyeon Choung, ohhyeon.choung@gmail.com
* Andreas Poehlmann, https://github.com/ap--
* Ya Chen, https://github.com/anya-chen
