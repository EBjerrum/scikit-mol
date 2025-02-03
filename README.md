# scikit-mol

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/EBjerrum/scikit-mol/blob/30c74b3648c0087bdb1b659bc67ba757d7498e9a/ressources/logo/ScikitMol_Logo_DarkBG_300px.png?raw=true">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/EBjerrum/scikit-mol/blob/30c74b3648c0087bdb1b659bc67ba757d7498e9a/ressources/logo/ScikitMol_Logo_LightBG_300px.png?raw=true">
  <img src="https://github.com/EBjerrum/scikit-mol/blob/30c74b3648c0087bdb1b659bc67ba757d7498e9a/ressources/logo/ScikitMol_Logo_LightBG_300px.png?raw=true" alt="Fancy logo">
</picture>

[![python versions](https://shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)]()

[![pypi version](https://img.shields.io/pypi/v/scikit-mol.svg)](https://pypi.org/project/scikit-mol/)
[![conda version](https://img.shields.io/conda/vn/conda-forge/scikit-mol.svg)](https://anaconda.org/conda-forge/scikit-mol)
[![license](https://img.shields.io/pypi/l/scikit-mol)](#)

[![powered by rdkit](https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAFVBMVEXc3NwUFP8UPP9kZP+MjP+0tP////9ZXZotAAAAAXRSTlMAQObYZgAAAAFiS0dEBmFmuH0AAAAHdElNRQfmAwsPGi+MyC9RAAAAQElEQVQI12NgQABGQUEBMENISUkRLKBsbGwEEhIyBgJFsICLC0iIUdnExcUZwnANQWfApKCK4doRBsKtQFgKAQC5Ww1JEHSEkAAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyMi0wMy0xMVQxNToyNjo0NyswMDowMDzr2J4AAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjItMDMtMTFUMTU6MjY6NDcrMDA6MDBNtmAiAAAAAElFTkSuQmCC)](https://www.rdkit.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Scikit-Learn classes for molecular vectorization using RDKit

The intended usage is to be able to add molecular vectorization directly into scikit-learn pipelines, so that the final model directly predict on RDKit molecules or SMILES strings

As example with the needed scikit-learn and -mol imports and RDKit mol objects in the mol_list_train and \_test lists:

    pipe = Pipeline([('mol_transformer', MorganFingerprintTransformer()), ('Regressor', Ridge())])
    pipe.fit(mol_list_train, y_train)
    pipe.score(mol_list_test, y_test)
    pipe.predict([Chem.MolFromSmiles('c1ccccc1C(=O)C')])

    >>> array([4.93858815])

The scikit-learn compatibility should also make it easier to include the fingerprinting step in hyperparameter tuning with scikit-learns utilities

The first draft for the project was created at the [RDKIT UGM 2022 hackathon](https://github.com/rdkit/UGM_2022) 2022-October-14

## Implemented

- Descriptors
  - MolecularDescriptorTransformer

<br>

- Fingerprints
  - MorganFingerprintTransformer
  - MACCSKeysFingerprintTransformer
  - RDKitFingerprintTransformer
  - AtomPairFingerprintTransformer
  - TopologicalTorsionFingerprintTransformer
  - MHFingerprintTransformer
  - SECFingerprintTransformer
  - AvalonFingerprintTransformer

<br>

- Conversions
  - SmilesToMol

<br>

- Standardizer
  - Standardizer

<br>

- safeinference
  - SafeInferenceWrapper
  - set_safe_inference_mode

<br>

- Utilities
  - CheckSmilesSanitazion

## Installation

Users can install latest tagged release from pip

    pip install scikit-mol

or from conda-forge

    conda install -c conda-forge scikit-mol

The conda forge package should get updated shortly after a new tagged release on pypi.

Bleeding edge

    pip install git+https://github.com:EBjerrum/scikit-mol.git

## Documentation

There are a collection of notebooks in the notebooks directory which demonstrates some different aspects and use cases

- [Basic Usage and fingerprint transformers](https://github.com/EBjerrum/scikit-mol/tree/main/notebooks/01_basic_usage.ipynb)
- [Descriptor transformer](https://github.com/EBjerrum/scikit-mol/tree/main/notebooks/02_descriptor_transformer.ipynb)
- [Pipelining with Scikit-Learn classes](https://github.com/EBjerrum/scikit-mol/tree/main/notebooks/03_example_pipeline.ipynb)
- [Molecular standardization](https://github.com/EBjerrum/scikit-mol/tree/main/notebooks/04_standardizer.ipynb)
- [Sanitizing SMILES input](https://github.com/EBjerrum/scikit-mol/tree/main/notebooks/05_smiles_sanitaztion.ipynb)
- [Integrated hyperparameter tuning of Scikit-Learn estimator and Scikit-Mol transformer](https://github.com/EBjerrum/scikit-mol/tree/main/notebooks/06_hyperparameter_tuning.ipynb)
- [Using parallel execution to speed up descriptor and fingerprint calculations](https://github.com/EBjerrum/scikit-mol/tree/main/notebooks/07_parallel_transforms.ipynb)
- [Using skopt for hyperparameter tuning](https://github.com/EBjerrum/scikit-mol/tree/main/notebooks/08_external_library_skopt.ipynb)
- [Testing different fingerprints as part of the hyperparameter optimization](https://github.com/EBjerrum/scikit-mol/blob/main/notebooks/09_Combinatorial_Method_Usage_with_FingerPrint_Transformers.ipynb)
- [Using pandas output for easy feature importance analysis and combine pre-exisitng values with new computations](https://github.com/EBjerrum/scikit-mol/blob/main/notebooks/10_pipeline_pandas_output.ipynb)
- [Working with pipelines and estimators in safe inference mode for handling prediction on batches with invalid smiles or molecules](https://github.com/EBjerrum/scikit-mol/blob/main/notebooks/11_safe_inference.ipynb)

  We also put a software note on ChemRxiv. [https://doi.org/10.26434/chemrxiv-2023-fzqwd](https://doi.org/10.26434/chemrxiv-2023-fzqwd)

## Other use-examples

Scikit-Mol has been featured in blog-posts or used in research, some examples which are listed below:

- [Useful ML package for cheminformatics iwatobipen.wordpress.com](https://iwatobipen.wordpress.com/2023/11/12/useful-ml-package-for-cheminformatics-rdkit-cheminformatics-ml/)
- [Boosted trees Data_in_life_blog](https://jhylin.github.io/Data_in_life_blog/posts/19_ML2-3_Boosted_trees/1_adaboost_xgb.html)
- [Konnektor: A Framework for Using Graph Theory to Plan Networks for Free Energy Calculations](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.4c01710)
- [Moldrug algorithm for an automated ligand binding site exploration by 3D aware molecular enumerations](https://chemrxiv.org/engage/chemrxiv/article-details/67688633fa469535b97c1b73)
- [RandomNets Improve Neural Network Regression Performance via Implicit Ensembling](https://chemrxiv.org/engage/chemrxiv/article-details/67656cfa81d2151a02603f48)
- [WAE-DTI: Ensemble-based architecture for drug–target interaction prediction using descriptors and embeddings](https://www.sciencedirect.com/science/article/pii/S2352914824001618)
- [Data Driven Estimation of Molecular Log-Likelihood using Fingerprint Key Counting](https://chemrxiv.org/engage/chemrxiv/article-details/661402ee21291e5d1d646651)
- [AUTONOMOUS DRUG DISCOVERY](https://www.proquest.com/openview/3e830e36bc618f263905a99e787c66c6/1?pq-origsite=gscholar&cbl=18750&diss=y)

## Roadmap and Contributing

_Help wanted!_ Are you a PhD student that want a "side-quest" to procrastinate your thesis writing or are you simply interested in computational chemistry, cheminformatics or simply with an interest in QSAR modelling, Python Programming open-source software? Do you want to learn more about machine learning with Scikit-Learn? Or do you use scikit-mol for your current work and would like to pay a little back to the project and see it improved as well?
With a little bit of help, this project can be improved much faster! Reach to me (Esben), for a discussion about how we can proceed.

Currently we are working on fixing some deprecation warnings, its not the most exciting work, but it's important to maintain a little. Later on we need to go over the scikit-learn compatibility and update to some of their newer features on their estimator classes. We're also brewing on some feature enhancements and tests, such as new fingerprints and a more versatile standardizer.

There are more information about how to contribute to the project in [CONTRIBUTING](CONTRIBUTING.md)

## BUGS

Probably still, please check issues at GitHub and report there

## Contributers:

- Esben Jannik Bjerrum [@ebjerrum](https://github.com/ebjerrum), esbenbjerrum+scikit_mol@gmail.com
- Carmen Esposito [@cespos](https://github.com/cespos)
- Son Ha, sonha@uni-mainz.de
- Oh-hyeon Choung, ohhyeon.choung@gmail.com
- Andreas Poehlmann, [@ap--](https://github.com/ap--)
- Ya Chen, [@anya-chen](https://github.com/anya-chen)
- Anton Siomchen [@asiomchen](https://github.com/asiomchen)
- Rafał Bachorz [@rafalbachorz](https://github.com/rafalbachorz)
- Adrien Chaton [@adrienchaton](https://github.com/adrienchaton)
- [@VincentAlexanderScholz](https://github.com/VincentAlexanderScholz)
- [@RiesBen](https://github.com/RiesBen)
- [@enricogandini](https://github.com/enricogandini)
- [@mikemhenry](https://github.com/mikemhenry)
- [@c-feldmann](https://github.com/c-feldmann)
