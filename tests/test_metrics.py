import numpy as np
import pytest

from scikit_mol.metrics import tanimoto_distance

from .applicability.conftest import binary_fingerprints


@pytest.fixture
def simple_fingerprints():
    """Create simple binary fingerprints for testing."""
    return np.array(
        [
            [1, 1, 0, 0],  # fp0: 2 bits set
            [1, 0, 1, 0],  # fp1: 2 bits set, 1 in common with fp1
            [0, 0, 1, 1],  # fp2: 2 bits set, 1 in common with fp2, none with fp1
            [1, 1, 1, 1],  # fp3: all bits set
            [0, 0, 0, 0],  # fp4: no bits set
        ],
        dtype=bool,
    )


def test_tanimoto_distance_basic(simple_fingerprints):
    """Test basic properties of Tanimoto distance."""
    distances = tanimoto_distance(simple_fingerprints[0], simple_fingerprints[1])

    # Check distance range [0,1]
    assert 0 <= distances <= 1

    # Check specific distances
    # fp0 vs fp1: 1 bit in common, 3 in union -> distance = 2/3
    assert np.isclose(distances, 2 / 3)
    # fp0 vs fp2: no bits in common, 4 in union -> distance = 1
    assert np.isclose(
        tanimoto_distance(simple_fingerprints[0], simple_fingerprints[2]), 1.0
    )
    # fp0 vs fp3: 2 bits in common, 4 in union -> distance = 0.5
    assert np.isclose(
        tanimoto_distance(simple_fingerprints[0], simple_fingerprints[3]), 0.5
    )
    # fp0 vs fp4: no bits in common, 2 in union -> distance = 1
    assert np.isclose(
        tanimoto_distance(simple_fingerprints[0], simple_fingerprints[4]), 1.0
    )


def test_tanimoto_distance_edge_cases(simple_fingerprints):
    """Test edge cases for Tanimoto distance."""
    empty = simple_fingerprints[4]  # Empty fingerprint
    full = simple_fingerprints[3]  # Full fingerprint

    # Two empty fingerprints (fp4)
    dist = tanimoto_distance(empty, empty)
    # No bits in common, 0 in union -> distance = 0/0 = 0 in our implementation.
    assert np.isclose(dist, 0.0)

    # Empty vs full fingerprint (fp3)
    dist = tanimoto_distance(empty, full)
    assert np.isclose(dist, 1.0)  # No overlap -> maximum distance


# TODO, can rdkit speed things up? But not working with np.arrays
# def test_tanimoto_implementations_equivalent(simple_fingerprints):
#     """Test that both implementations give equivalent results."""
#     X = simple_fingerprints[:2]
#     Y = simple_fingerprints[2:4]

#     dist1 = tanimoto_distance(X, Y)
#     dist2 = tanimoto_distance_rdkit(X, Y)

#     assert np.allclose(dist1, dist2)


# def test_tanimoto_distance_rdkit_basic(binary_fingerprints):
#     """Test basic properties of RDKit-based Tanimoto distance."""
#     # Get a subset of fingerprints for testing
#     X = binary_fingerprints[:3]
#     Y = binary_fingerprints[3:6]

#     distances = tanimoto_distance_rdkit(X, Y)

#     # Check output shape
#     assert distances.shape == (3, 3)

#     # Check distance range [0,1]
#     assert np.all((0 <= distances) & (distances <= 1))

#     # Check self-distance is 0 for identical fingerprints
#     self_distances = tanimoto_distance_rdkit(X, X)
#     assert np.allclose(np.diag(self_distances), 0)
