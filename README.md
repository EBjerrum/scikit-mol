# scikit-mol

![Scikit-Mol Logo](https://raw.githubusercontent.com/EBjerrum/scikit-mol/029036fed8575705eaa80f6e3b08e70463b9a0c4/resources/logo/ScikitMol_Logo_Hybrid_300.png)

[![python versions](https://shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)]()

[![pypi version](https://img.shields.io/pypi/v/scikit-mol.svg)](https://pypi.org/project/scikit-mol/)
[![conda version](https://img.shields.io/conda/vn/conda-forge/scikit-mol.svg)](https://anaconda.org/conda-forge/scikit-mol)
[![license](https://img.shields.io/github/license/EBjerrum/scikit-mol)](#)

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

## Installation

Users can install latest tagged release from pip

```sh
pip install scikit-mol
```

or from conda-forge

```sh
conda install -c conda-forge scikit-mol
```

The conda forge package should get updated shortly after a new tagged release on pypi.

Bleeding edge

```sh
pip install git+https://github.com/EBjerrum/scikit-mol.git
```

## Documentation

Example notebooks and API documentation are now hosted on [https://scikit-mol.readthedocs.io](https://scikit-mol.readthedocs.io/en/latest/)

- [Basic Usage and fingerprint transformers](https://scikit-mol.readthedocs.io/en/latest/notebooks/01_basic_usage/)
- [Descriptor transformer](https://scikit-mol.readthedocs.io/en/latest/notebooks/02_descriptor_transformer/)
- [Pipelining with Scikit-Learn classes](https://scikit-mol.readthedocs.io/en/latest/notebooks/03_example_pipeline/)
- [Molecular standardization](https://scikit-mol.readthedocs.io/en/latest/notebooks/04_standardizer/)
- [Sanitizing SMILES input](https://scikit-mol.readthedocs.io/en/latest/notebooks/05_smiles_sanitization/)
- [Integrated hyperparameter tuning of Scikit-Learn estimator and Scikit-Mol transformer](https://scikit-mol.readthedocs.io/en/latest/notebooks/06_hyperparameter_tuning/)
- [Using parallel execution to speed up descriptor and fingerprint calculations](https://scikit-mol.readthedocs.io/en/latest/notebooks/07_parallel_transforms/)
- [Using skopt for hyperparameter tuning](https://scikit-mol.readthedocs.io/en/latest/notebooks/08_external_library_skopt/)
- [Testing different fingerprints as part of the hyperparameter optimization](https://scikit-mol.readthedocs.io/en/latest/notebooks/09_Combinatorial_Method_Usage_with_FingerPrint_Transformers/)
- [Using pandas output for easy feature importance analysis and combine pre-existing values with new computations](https://scikit-mol.readthedocs.io/en/latest/notebooks/10_pipeline_pandas_output/)
- [Working with pipelines and estimators in safe inference mode for handling prediction on batches with invalid smiles or molecules](https://scikit-mol.readthedocs.io/en/latest/notebooks/11_safe_inference/)
- [Creating custom fingerprint transformers](https://scikit-mol.readthedocs.io/en/latest/notebooks/12_custom_fingerprint_transformer/)
- [Estimating applicability domain using feature based estimators](https://scikit-mol.readthedocs.io/en/latest/notebooks/13_applicability_domain/)

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
- [DrugGym: A testbed for the economics of autonomous drug discovery](https://www.biorxiv.org/content/10.1101/2024.05.28.596296v1.abstract)

## Roadmap and Contributing

_Help wanted!_ Are you a PhD student that want a "side-quest" to procrastinate your thesis writing or are you simply interested in computational chemistry, cheminformatics or simply with an interest in QSAR modelling, Python Programming open-source software? Do you want to learn more about machine learning with Scikit-Learn? Or do you use scikit-mol for your current work and would like to pay a little back to the project and see it improved as well?
With a little bit of help, this project can be improved much faster! Reach to me (Esben), for a discussion about how we can proceed.

Currently, we are working on fixing some deprecation warnings, it's not the most exciting work, but it's important to maintain a little. Later on we need to go over the scikit-learn compatibility and update to some of their newer features on their estimator classes. We're also brewing on some feature enhancements and tests, such as new fingerprints and a more versatile standardizer.

There are more information about how to contribute to the project in [CONTRIBUTING](https://scikit-mol.readthedocs.io/en/latest/contributing/)

## BUGS

Probably still, please check issues at GitHub and report there

## Contributors

Scikit-Mol has been developed as a community effort with contributions from people from many different companies, consortia, foundations and academic institutions.

[Cheminformania Consulting](https://www.cheminformania.com), [Aptuit](https://www.linkedin.com/company/aptuit/), [BASF](https://www.basf.com), [Bayer AG](https://www.bayer.com), [Boehringer Ingelheim](https://www.boehringer-ingelheim.com/), [Chodera Lab (MSKCC)](https://www.choderalab.org/), [EPAM Systems](https://www.epam.com/),[ETH Zürich](https://ethz.ch/en.html), [Evotec](https://www.evotec.com/), [Johannes Gutenberg University](https://www.uni-mainz.de/en/), [Martin Luther University](https://www.uni-halle.de/?lang=en), [Odyssey Therapeutics](https://odysseytx.com/), [Open Molecular Software Foundation](https://omsf.io/), [Openfree.energy](https://openfree.energy/), [Polish Academy of Sciences](https://pasific.pan.pl/polish-academy-of-sciences/), [Productivista](https://www.productivista.com), [Simulations-Plus Inc.](https://www.simulations-plus.com/), [University of Vienna](https://www.univie.ac.at/en/)

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
