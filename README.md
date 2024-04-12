# scikit-mol
![Fancy logo](./ressources/logo/ScikitMol_Logo_DarkBG_300px.png#gh-dark-mode-only)
![Fancy logo](./ressources/logo/ScikitMol_Logo_LightBG_300px.png#gh-light-mode-only)
## Scikit-Learn classes for molecular vectorization using RDKit

The intended usage is to be able to add molecular vectorization directly into scikit-learn pipelines, so that the final model directly predict on RDKit molecules or SMILES strings

As example with the needed scikit-learn and -mol imports and RDKit mol objects in the mol_list_train and _test lists:

    pipe = Pipeline([('mol_transformer', MorganFingerprintTransformer()), ('Regressor', Ridge())])
    pipe.fit(mol_list_train, y_train)
    pipe.score(mol_list_test, y_test)
    pipe.predict([Chem.MolFromSmiles('c1ccccc1C(=O)C')])

    >>> array([4.93858815])

The scikit-learn compatibility should also make it easier to include the fingerprinting step in hyperparameter tuning with scikit-learns utilities

The first draft for the project was created at the [RDKIT UGM 2022 hackathon](https://github.com/rdkit/UGM_2022) 2022-October-14


## Implemented
* Descriptors
    * MolecularDescriptorTransformer

<br>

* Fingerprints
    * MorganFingerprintTransformer
    * MACCSKeysFingerprintTransformer
    * RDKitFingerprintTransformer
    * AtomPairFingerprintTransformer
    * TopologicalTorsionFingerprintTransformer
    * MHFingerprintTransformer
    * SECFingerprintTransformer
    * AvalonFingerprintTransformer

<br>

* Conversions
    * SmilesToMol

<br>

* Standardizer
    * Standardizer

<br>

* Utilities
    * CheckSmilesSanitazion

## Installation
Users can install latest tagged release from pip

    pip install scikit-mol

Bleeding edge

    pip install git+https://github.com:EBjerrum/scikit-mol.git

## Documentation

There are a collection of notebooks in the notebooks directory which demonstrates some different aspects and use cases

* [Basic Usage and fingerprint transformers](https://github.com/EBjerrum/scikit-mol/blob/main/notebooks/01_basic_usage.ipynb)
* [Descriptor transformer](https://github.com/EBjerrum/scikit-mol/blob/main/notebooks/02_descriptor_transformer.ipynb)
* [Pipelining with Scikit-Learn classes](https://github.com/EBjerrum/scikit-mol/blob/main/notebooks/03_example_pipeline.ipynb)
* [Molecular standardization](https://github.com/EBjerrum/scikit-mol/blob/main/notebooks/04_standardizer.ipynb)
* [Sanitizing SMILES input](https://github.com/EBjerrum/scikit-mol/blob/main/notebooks/05_smiles_sanitaztion.ipynb)
* [Integrated hyperparameter tuning of Scikit-Learn estimator and Scikit-Mol transformer](https://github.com/EBjerrum/scikit-mol/blob/main/notebooks/06_hyperparameter_tuning.ipynb)
* [Using parallel execution to speed up descriptor and fingerprint calculations](https://github.com/EBjerrum/scikit-mol/blob/main/notebooks/07_parallel_transforms.ipynb)
* [Testing different fingerprints as part of the hyperparameter optimization](https://github.com/EBjerrum/scikit-mol/blob/main/notebooks/09_Combinatorial_Method_Usage_with_FingerPrint_Transformers.ipynb)
* [Using pandas output for easy feature importance analysis and combine pre-exisitng values with new computations](https://github.com/EBjerrum/scikit-mol/blob/main/notebooks/10_pipeline_pandas_output.ipynb)


  We also put a software note on ChemRxiv. [https://doi.org/10.26434/chemrxiv-2023-fzqwd](https://doi.org/10.26434/chemrxiv-2023-fzqwd) 

## Contributing

There are more information about how to contribute to the project in [CONTRIBUTION.md](https://github.com/EBjerrum/scikit-mol/CONTRIBUTION.md)

## BUGS
Probably still, please check issues at GitHub and report there

## Contributers:
* Esben Jannik Bjerrum [@ebjerrum](https://github.com/ebjerrum), esbenbjerrum+scikit_mol@gmail.com
* Carmen Esposito [@cespos](https://github.com/cespos)
* Son Ha, sonha@uni-mainz.de
* Oh-hyeon Choung, ohhyeon.choung@gmail.com
* Andreas Poehlmann, [@ap--](https://github.com/ap--)
* Ya Chen, [@anya-chen](https://github.com/anya-chen)
* Rafał Bachorz [@rafalbachorz](https://github.com/rafalbachorz)
* Adrien Chaton [@adrienchaton](https://github.com/adrienchaton)
* [@VincentAlexanderScholz](https://github.com/VincentAlexanderScholz)
* [@RiesBen](https://github.com/RiesBen)
* [@enricogandini](https://github.com/enricogandini)
