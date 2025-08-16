from collections.abc import Sequence
from typing import Optional, Union
from abc import ABC, abstractmethod
import rdkit

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.rdBase import BlockLogs
from sklearn.base import BaseEstimator, TransformerMixin
from rdkit.Chem.Scaffolds import MurckoScaffold

from scikit_mol._constants import DOCS_BASE_URL
from scikit_mol.core import (
    InvalidMol,
    NoFitNeededMixin,
    check_transform_input,
    feature_names_default_mol,
)
from scikit_mol.parallel import parallelized_with_batches

# from scikit_mol._invalid import InvalidMol


class SmilesToMolTransformer(TransformerMixin, NoFitNeededMixin, BaseEstimator):
    """
    Transformer for converting SMILES strings to RDKit mol objects.

    This transformer can be included in pipelines during development and training,
    but the safe inference mode should only be enabled when deploying models for
    inference in production environments.
    """

    _doc_link_module = "scikit_mol"
    _doc_link_template = (
        DOCS_BASE_URL + "{estimator_module}/#{estimator_module}.{estimator_name}"
    )

    def __init__(
        self, n_jobs: Optional[None] = None, safe_inference_mode: bool = False
    ):
        """
        Parameters
        -----------
        n_jobs : int, optional default=None
            The maximum number of concurrently running jobs.
            `None` is a marker for 'unset' that will be interpreted as `n_jobs=1` unless the call is performed under a `parallel_config()` context manager that sets another value for `n_jobs`.
        safe_inference_mode : bool, default=False
            If `True`, enables safeguards for handling invalid data during inference.
            This should only be set to `True` when deploying models to production.
        """
        self.n_jobs = n_jobs
        self.safe_inference_mode = safe_inference_mode

    @feature_names_default_mol
    def get_feature_names_out(self, input_features=None):
        return input_features

    def fit(self, X=None, y=None):
        """Included for scikit-learn compatibility, does nothing"""
        return self

    def transform(
        self, X_smiles_list: Sequence[str], y=None
    ) -> NDArray[Union[Chem.Mol, InvalidMol]]:
        """Converts SMILES into RDKit mols

        Parameters
        ----------
        X_smiles_list : Sequence[str]
            sequence of SMILES strings to transform

        Returns
        -------
        NDArray[Union[Chem.Mol, InvalidMol]]
            Array of RDKit mol objects or InvalidMol objects if a SMILES string is invalid and `safe_inference_mode=True`
        Raises
        ------
        ValueError
            Raises ValueError if a SMILES string is unparsable by RDKit and `safe_inference_mode=False`
        """

        arrays = parallelized_with_batches(self._transform, X_smiles_list, self.n_jobs)
        arr = np.concatenate(arrays)
        return arr

    @check_transform_input
    def _transform(self, X):
        X_out = []
        with BlockLogs():
            for smiles in X:
                mol = Chem.MolFromSmiles(smiles, sanitize=False)
                if mol:
                    errors = Chem.DetectChemistryProblems(mol)
                    if errors:
                        error_message = "\n".join(error.Message() for error in errors)
                        message = f"Invalid Molecule: {error_message}"
                        X_out.append(InvalidMol(str(self), message))
                    else:
                        Chem.SanitizeMol(mol)
                        X_out.append(mol)
                else:
                    message = f"Invalid SMILES: {smiles}"
                    X_out.append(InvalidMol(str(self), message))
        if not self.safe_inference_mode and not all(X_out):
            fails = [x for x in X_out if not x]
            raise ValueError(
                f"Invalid input found: {fails}."
            )  # TODO with this approach we get all errors, but we do process ALL the smiles first which could be slow
        return np.array(X_out).reshape(-1, 1)

    @check_transform_input
    def inverse_transform(self, X_mols_list, y=None):
        X_out = []

        for mol in X_mols_list:
            if isinstance(mol, Chem.Mol):
                try:
                    smiles = Chem.MolToSmiles(mol)
                    X_out.append(smiles)
                except Exception as e:
                    X_out.append(
                        InvalidMol(str(self), f"Error converting Mol to SMILES: {e}")
                    )
            else:
                X_out.append(InvalidMol(str(self), f"Not a Mol: {mol}"))

        if not self.safe_inference_mode and not all(isinstance(x, str) for x in X_out):
            fails = [x for x in X_out if not isinstance(x, str)]
            raise ValueError(f"Invalid Mols found: {fails}.")

        return np.array(X_out).reshape(-1, 1)


class BaseScaffoldGenerator(BaseEstimator, ABC):
    """
    Abstract base class for scaffold generators.
    Inherits from scikit-learn's BaseEstimator to allow for
    integration into pipelines and hyperparameter tuning.
    """

    @abstractmethod
    def get_scaffold(self, mol: Chem.Mol) -> Chem.Mol:
        """
        Generates a scaffold for a given RDKit molecule.
        Parameters
        ----------
        mol : Chem.Mol
            The input molecule.
        Returns
        -------
        Chem.Mol
            The resulting scaffold molecule. Can be None if generation fails.
        """
        raise NotImplementedError


class MurckoScaffoldGenerator(BaseScaffoldGenerator):
    """
    Generates Murcko scaffolds for molecules.
    This class encapsulates the logic for RDKit's Murcko scaffold
    functionality and serves as a concrete implementation of the
    BaseScaffoldGenerator.
    Parameters
    ----------
    include_chirality : bool, default=False
        Whether to include chirality in the scaffold. This is passed to
        `MurckoScaffold.MurckoScaffoldSmiles`.
    make_generic : bool, default=False
        Whether to convert the scaffold to a generic representation
        (all atoms become carbons, all bonds become single).
    """

    def __init__(self, include_chirality: bool = False, make_generic: bool = False):
        self.include_chirality = include_chirality
        self.make_generic = make_generic

    def get_scaffold(self, mol: Chem.Mol) -> Chem.Mol:
        """Generates a Murcko scaffold for the input molecule."""
        # We always use MurckoScaffoldSmiles to have explicit control over chirality.
        scaffold_smiles = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=self.include_chirality
        )

        # An empty SMILES string is returned for acyclic molecules.
        # Chem.MolFromSmiles("") returns None, which is the desired output for these cases.
        scaffold_mol = Chem.MolFromSmiles(scaffold_smiles) if scaffold_smiles else None

        if self.make_generic and scaffold_mol:
            scaffold_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)

        return scaffold_mol


class MolToScaffoldTransformer(SmilesToMolTransformer):
    """
    Transformer for converting RDKit Mol objects to molecular scaffolds.
    This transformer assumes the input is already a sequence of Mol objects.
    """

    def __init__(
        self,
        scaffold_generator: Optional[BaseScaffoldGenerator] = None,
        n_jobs: Optional[int] = None,
        safe_inference_mode: bool = False,
    ):
        """
        Parameters
        ----------
        scaffold_generator : object, optional
            An object with a `get_scaffold(mol)` method that returns an RDKit Mol object.
            If None, a default `MurckoScaffoldGenerator()` is used.
        n_jobs : int, optional
            The maximum number of concurrently running jobs.
            `None` is a marker for 'unset' that will be interpreted as `n_jobs=1`
            unless the call is performed under a `parallel_config()` context manager
            that sets another value for `n_jobs`.
        safe_inference_mode : bool, default=False
            If `True`, enables safeguards for handling invalid data during inference.
            This should only be set to `True` when deploying models to production.
        """
        # Call the parent's __init__ to handle shared parameters like n_jobs
        # and safe_inference_mode.
        super().__init__(
            n_jobs=n_jobs, safe_inference_mode=safe_inference_mode
        )

        if scaffold_generator is None:
            self.scaffold_generator = MurckoScaffoldGenerator()
        else:
            if not isinstance(scaffold_generator, BaseScaffoldGenerator):
                raise TypeError(
                    "scaffold_generator must be an instance of BaseScaffoldGenerator."
                )
            self.scaffold_generator = scaffold_generator

    def transform(
        self, X_mols_list: Sequence[Union[Chem.Mol, InvalidMol]], y=None
    ) -> NDArray[Union[Chem.Mol, InvalidMol]]:
        """
        Converts RDKit Mol objects into molecular scaffolds.
        Parameters
        ----------
        X_mols_list : Sequence[Union[Chem.Mol, InvalidMol]]
            A sequence of RDKit Mol objects or InvalidMol objects.
        Returns
        -------
        NDArray[Union[Chem.Mol, InvalidMol]]
            An array of scaffold molecules or InvalidMol objects.
        """
        # The input is expected to be a list/array of Mol objects.
        # We flatten the input in case it's a 2D array from a previous transformer.
        mols = np.array(X_mols_list).flatten()

        # Parallelize the scaffold generation for efficiency
        scaffold_arrays = parallelized_with_batches(
            self._scaffold_transform, mols, self.n_jobs
        )
        scaffolds = np.concatenate(scaffold_arrays)

        return scaffolds.reshape(-1, 1)

    def _scaffold_transform(self, mols: Sequence[Union[Chem.Mol, InvalidMol]]):
        scaffolds = []
        for mol in mols:
            if isinstance(mol, Chem.Mol):
                try:
                    scaffold = self.scaffold_generator.get_scaffold(mol)
                    if scaffold is None:
                        # If scaffold generation results in None (e.g., for acyclic molecules),
                        # we create an InvalidMol object to signify this.
                        scaffolds.append(
                            InvalidMol(str(self), "Scaffold generation resulted in an empty molecule.")
                        )
                    else:
                        scaffolds.append(scaffold)
                except Exception as e:
                    scaffolds.append(
                        InvalidMol(str(self), f"Error creating scaffold: {e}")
                    )
            else:
                scaffolds.append(mol)  # Pass through InvalidMol objects
        return np.array(scaffolds)