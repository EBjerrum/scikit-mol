import hashlib
import shutil
from pathlib import Path, PurePath
from urllib.parse import urlsplit
from urllib.request import urlopen

import numpy as np
import pandas as pd
import pytest
import sklearn


# Register custom marks
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "threshold_fitting: mark tests that verify threshold fitting functionality",
    )


TEST_DATA_URL = "https://ndownloader.figshare.com/files/25747817"
TEST_DATA_MD5 = "1ec89bde544c3c4bc400d5b75315921e"


def md5(fn):
    m = hashlib.md5()
    with open(fn, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            m.update(chunk)
    return m.hexdigest()


@pytest.fixture(scope="session")
def data_pth(tmp_path_factory) -> Path:
    """download the smallest aperio test image svs or use local"""
    filename = PurePath(urlsplit(TEST_DATA_URL).path).name
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_fn = data_dir / filename

    if not data_fn.is_file():
        # download svs from openslide test images
        with urlopen(TEST_DATA_URL) as response, open(data_fn, "wb") as out_file:
            shutil.copyfileobj(response, out_file)

        if md5(data_fn) != TEST_DATA_MD5:  # pragma: no cover
            shutil.rmtree(data_fn)
            pytest.fail("incorrect md5")

    yield data_fn.absolute()


@pytest.fixture()
def data(data_pth) -> pd.DataFrame:
    yield pd.read_csv(data_pth)


@pytest.fixture(scope="module")
def pandas_output():
    """Set sklearn to output pandas dataframes"""
    sklearn.set_config(transform_output="pandas")
    yield
    sklearn.set_config(transform_output="default")


# Fixed Numpy random seed in all tests automatically
@pytest.fixture(autouse=True)
def setup_random():
    """Set fixed random seed before each test."""
    np.random.seed(0xDEADFACE)
