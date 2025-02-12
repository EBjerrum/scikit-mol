from scikit_mol._constants import DOCS_BASE_URL

from .atompair import AtomPairFingerprintTransformer
from .avalon import AvalonFingerprintTransformer

# TODO, these baseclasses needed for backwards compatibility with tests, needs to be removed when tests updated
from .baseclasses import (
    FpsGeneratorTransformer,
    FpsTransformer,
)
from .maccs import MACCSKeysFingerprintTransformer
from .minhash import MHFingerprintTransformer, SECFingerprintTransformer
from .morgan import MorganFingerprintTransformer
from .rdkitfp import RDKitFingerprintTransformer
from .topologicaltorsion import (
    TopologicalTorsionFingerprintTransformer,
)

__all__ = [
    "AtomPairFingerprintTransformer",
    "AvalonFingerprintTransformer",
    "FpsGeneratorTransformer",
    "FpsTransformer",
    "MACCSKeysFingerprintTransformer",
    "MHFingerprintTransformer",
    "MorganFingerprintTransformer",
    "RDKitFingerprintTransformer",
    "SECFingerprintTransformer",
    "TopologicalTorsionFingerprintTransformer",
]

for name in __all__:
    if name.startswith("Fps"):
        continue
    cls = locals()[name]
    cls._doc_link_module = "scikit_mol"
    cls._doc_link_template = (
        DOCS_BASE_URL
        + "scikit_mol.fingerprints/#scikit_mol.fingerprints.{estimator_name}"
    )
