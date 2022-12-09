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
    * SECFingerprintTransformer
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

There are a collection of notebooks in the notebooks directory which demonstrates some different aspects and use cases

* [Basic Usage and fingerprint transformers](https://github.com/EBjerrum/scikit-mol/blob/documentation/notebooks/01_basic_usage.ipynb)
* [Descriptor transformer](https://github.com/EBjerrum/scikit-mol/blob/documentation/notebooks/02_descriptor_transformer.ipynb)
* [Pipelining with Scikit-Learn classes](https://github.com/EBjerrum/scikit-mol/blob/documentation/notebooks/03_example_pipeline.ipynb)
* [Molecular standardization](https://github.com/EBjerrum/scikit-mol/blob/documentation/notebooks/04_standardizer.ipynb)
* [Sanitizing SMILES input](https://github.com/EBjerrum/scikit-mol/blob/documentation/notebooks/05_smiles_sanitaztion.ipynb)
* [Integrated hyperparameter tuning of Scikit-Learn estimator and Scikit-Mol transformer](https://github.com/EBjerrum/scikit-mol/blob/documentation/notebooks/06_hyperparameter_tuning.ipynb)


## BUGS
Probably still

## Contributers:
* Esben Jannik Bjerrum [@ebjerrum](https://github.com/ebjerrum), esben@cheminformania.com
* Carmen Esposito [@cespos](https://github.com/cespos)
* Son Ha, sonha@uni-mainz.de
* Oh-hyeon Choung, ohhyeon.choung@gmail.com
* Andreas Poehlmann, [@ap--](https://github.com/ap--)
* Ya Chen, [@anya-chen](https://github.com/anya-chen)
* Rafał Bachorz [@rafalbachorz](https://github.com/rafalbachorz)
