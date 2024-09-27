from typing import Optional, Sequence
from sklearn.base import BaseEstimator, TransformerMixin
from rdkit import Chem

from scikit_mol._invalid import (
    InvalidMol,
    rdkit_error_handling,
)


class TestInvalidTransformer(BaseEstimator, TransformerMixin):
    """This class is ment for tesing purposes only.

    All molecules with element number are returned as invalid instance.

    Attributes
    ---------
    atomic_number_set: set[int]
        Atomic numbers which upon occurrence in the molecule make it invalid.
    """

    atomic_number_set: set[int]

    def __init__(self, atomic_number_set: Sequence[int] | None = None) -> None:
        if atomic_number_set is None:
            atomic_number_set = {16}
        self.atomic_number_set = set(atomic_number_set)

    def _transform_mol(self, mol: Chem.Mol) -> Chem.Mol | InvalidMol:
        unique_elements = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
        forbidden_elements = self.atomic_number_set & unique_elements
        if forbidden_elements:
            return InvalidMol(str(self), f"Molecule contains {forbidden_elements}")
        return mol

    @rdkit_error_handling
    def transform(self, X: list[Chem.Mol]) -> list[Chem.Mol | InvalidMol]:
        return [self._transform_mol(mol) for mol in X]

    def fit(self, X, y, fit_params):
        pass

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)
