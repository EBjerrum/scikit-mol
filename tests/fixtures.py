import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from packaging.version import Version
import sklearn
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector, make_column_transformer
from scikit_mol.fingerprints import (
    MACCSKeysFingerprintTransformer,
    RDKitFingerprintTransformer,
    AtomPairFingerprintTransformer,
    TopologicalTorsionFingerprintTransformer,
    MorganFingerprintTransformer,
    SECFingerprintTransformer,
    MHFingerprintTransformer,
    AvalonFingerprintTransformer,
)
from scikit_mol.descriptors import MolecularDescriptorTransformer
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.standardizer import Standardizer
from scikit_mol.core import (
    SKLEARN_VERSION_PANDAS_OUT,
    DEFAULT_MOL_COLUMN_NAME,
    InvalidMol,
)

# TODO these should really go into the conftest.py, so that they are automatically imported in the tests

_SMILES_LIST = [
    "O=C(O)c1ccccc1",
    "O=C([O-])c1ccccc1",
    "O=C([O-])c1ccccc1.[Na+]",
    "O=C(O[Na])c1ccccc1",
    "C[N+](C)C.O=C([O-])c1ccccc1",
]
_CANONICAL_SMILES_LIST = [
    Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in _SMILES_LIST
]


@pytest.fixture
def smiles_list():
    return _CANONICAL_SMILES_LIST.copy()


_CONTAINER_CREATORS = [
    lambda x: x,
    lambda x: np.array(x),
    lambda x: np.array(x).reshape(-1, 1),
]
_names_to_test = [
    "molecule",
    "mol",
    "smiles",
    DEFAULT_MOL_COLUMN_NAME,
    "hello",
    None,
]
for name in _names_to_test:
    _CONTAINER_CREATORS.append(lambda x, name=name: pd.Series(x, name=name))
    _CONTAINER_CREATORS.append(
        lambda x, name=name: pd.DataFrame({name: x}) if name else pd.DataFrame(x)
    )


@pytest.fixture(
    params=[container(_CANONICAL_SMILES_LIST) for container in _CONTAINER_CREATORS]
)
def smiles_container(
    request,
):
    return request.param.copy()


@pytest.fixture
def chiral_smiles_list():  # Need to be a certain size, so the fingerprints reacts to different max_lenǵths and radii
    return [
        Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        for smiles in [
            "N[C@@H](C)C(=O)OCCCCCCCCCCCC",
            "C1C[C@H]2CCCC[C@H]2CC1CCCCCCCCC",
            "N[C@@H](C)C(=O)Oc1ccccc1CCCCCCCCCCCCCCCCCCN[H]",
        ]
    ]


@pytest.fixture
def invalid_smiles_list(smiles_list):
    smiles_list = smiles_list.copy()
    smiles_list.append("Invalid")
    return smiles_list


_MOLS_LIST = [Chem.MolFromSmiles(smiles) for smiles in _SMILES_LIST]


@pytest.fixture
def mols_list():
    return _MOLS_LIST.copy()


@pytest.fixture(params=[container(_MOLS_LIST) for container in _CONTAINER_CREATORS])
def mols_container(request):
    return request.param.copy()


@pytest.fixture
def chiral_mols_list(chiral_smiles_list):
    return [Chem.MolFromSmiles(smiles) for smiles in chiral_smiles_list]


@pytest.fixture
def mols_with_invalid_container(invalid_smiles_list):
    mols = []
    for smiles in invalid_smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mols.append(InvalidMol("TestError", f"Invalid SMILES: {smiles}"))
        else:
            mols.append(mol)
    return mols


@pytest.fixture
def fingerprint(mols_list):
    return rdMolDescriptors.GetHashedMorganFingerprint(mols_list[0], 2, nBits=1000)


_DIR_DATA = Path(__file__).parent / "data"
_FILE_SLC6A4 = _DIR_DATA / "SLC6A4_active_excapedb_subset.csv"
_FILE_SLC6A4_WITH_CDDD = _DIR_DATA / "CDDD_SLC6A4_active_excapedb_subset.csv.gz"


@pytest.fixture
def SLC6A4_subset():
    data = pd.read_csv(_FILE_SLC6A4)
    return data


@pytest.fixture
def SLC6A4_subset_with_cddd(SLC6A4_subset):
    data = SLC6A4_subset.copy().drop_duplicates(subset="Ambit_InchiKey")
    cddd = pd.read_csv(_FILE_SLC6A4_WITH_CDDD, index_col="Ambit_InchiKey")
    data = data.merge(
        cddd,
        left_on="Ambit_InchiKey",
        right_index=True,
        how="inner",
        validate="one_to_one",
    )
    return data


skip_pandas_output_test = pytest.mark.skipif(
    Version(sklearn.__version__) < SKLEARN_VERSION_PANDAS_OUT,
    reason=f"requires scikit-learn {SKLEARN_VERSION_PANDAS_OUT} or higher",
)

_FEATURIZER_CLASSES = [
    MACCSKeysFingerprintTransformer,
    RDKitFingerprintTransformer,
    AtomPairFingerprintTransformer,
    TopologicalTorsionFingerprintTransformer,
    MorganFingerprintTransformer,
    SECFingerprintTransformer,
    MHFingerprintTransformer,
    AvalonFingerprintTransformer,
    MolecularDescriptorTransformer,
]


@pytest.fixture(params=_FEATURIZER_CLASSES)
def featurizer(request):
    return request.param()


@pytest.fixture
def combined_transformer(featurizer):
    descriptors_pipeline = make_pipeline(
        SmilesToMolTransformer(),
        Standardizer(),
        featurizer,
    )
    # A pipeline that just passes the input data.
    # We will use it to preserve the CDDD features and pass them to downstream steps.
    identity_pipeline = make_pipeline(
        FunctionTransformer(),
    )
    transformer = make_column_transformer(
        (descriptors_pipeline, make_column_selector(pattern="SMILES")),
        (identity_pipeline, make_column_selector(pattern=r"^cddd_\d+$")),
        remainder="drop",
    )
    return transformer
