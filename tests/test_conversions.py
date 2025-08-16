"""
Tests for the scaffold generator classes in scikit_mol.conversions.
"""

import pytest
from rdkit import Chem
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from scikit_mol.conversions import (
    MurckoScaffoldGenerator,
    MolToScaffoldTransformer,
    SmilesToMolTransformer,
)

# A selection of SMILES strings for testing
MURCKO_TEST_CASES = {
    # Aspirin: simple case with a benzene ring and two side chains
    "CC(=O)OC1=CC=CC=C1C(=O)O": "c1ccccc1",
    # Ibuprofen: another common drug example
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O": "c1ccccc1",
    # Atorvastatin (Lipitor): complex molecule with multiple rings.
    # The expected scaffold is the result from RDKit's MurckoScaffold.
    "CC(C)C1=C(C(=C(N1C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=C(C=C4)F)C(O)CC(O)CC(=O)O": "O=C(Nc1ccccc1)c1ccn(-c2ccccc2)c1-c1ccccc1",
    # A molecule with no rings, should result in an empty scaffold
    "CCO": "",
    # A molecule with a chiral center but no rings, should also be empty
    "C[C@H](O)C(=O)O": "",  # Lactic acid
}


@pytest.fixture
def murcko_generator():
    """Provides a default MurckoScaffoldGenerator instance."""
    return MurckoScaffoldGenerator()


def test_murcko_generator_initialization():
    """
    Tests that the MurckoScaffoldGenerator initializes correctly and that
    its parameters are stored as public attributes, which is a requirement
    for scikit-learn compatibility.
    """
    generator = MurckoScaffoldGenerator(include_chirality=True, make_generic=True)
    assert generator.include_chirality
    assert generator.make_generic


def test_murcko_scaffold_generation(murcko_generator):
    """
    Tests the basic scaffold generation for a set of common molecules.
    This test ensures that the core functionality of converting a molecule
    to its Murcko scaffold is working as expected.
    """
    for smiles, expected_scaffold_smiles in MURCKO_TEST_CASES.items():
        mol = Chem.MolFromSmiles(smiles)
        scaffold = murcko_generator.get_scaffold(mol)

        # For molecules without rings, the scaffold should be ''
        if not expected_scaffold_smiles:
            assert scaffold is None, f"Scaffold should be empty for {smiles}"
            continue

        assert scaffold is not None, f"Scaffold generation failed for {smiles}"
        # Compare the SMILES representation of the generated scaffold to the expected one
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        assert scaffold_smiles == expected_scaffold_smiles, f"Incorrect scaffold for {smiles}"


def test_murcko_chirality_option():
    """
    Tests the `include_chirality` parameter of the MurckoScaffoldGenerator.
    When this option is enabled, the scaffold should preserve the stereochemistry
    of the original molecule. This test uses a molecule with a chiral ring,
    as Murcko scaffolds are only generated for cyclic systems.
    """
    # A chiral molecule with a ring system
    chiral_smiles = "C1CC[C@H]2CCCC[C@H]2C1"
    mol = Chem.MolFromSmiles(chiral_smiles)
    assert mol is not None, "Failed to create molecule from SMILES"

    # Test with chirality disabled (default)
    generator_no_chirality = MurckoScaffoldGenerator(include_chirality=False)
    scaffold_no_chirality = generator_no_chirality.get_scaffold(mol)
    assert scaffold_no_chirality is not None, "Scaffold should not be None"
    assert "@" not in Chem.MolToSmiles(
        scaffold_no_chirality
    ), "Scaffold should not have chiral centers"

    # Test with chirality enabled
    generator_with_chirality = MurckoScaffoldGenerator(include_chirality=True)
    scaffold_with_chirality = generator_with_chirality.get_scaffold(mol)
    assert scaffold_with_chirality is not None, "Chiral scaffold should not be None"
    assert "@" in Chem.MolToSmiles(
        scaffold_with_chirality
    ), "Scaffold should have chiral centers"


def test_murcko_generic_option():
    """
    Tests the `make_generic` parameter of the MurckoScaffoldGenerator.
    When this option is enabled, all atoms in the scaffold are converted to
    carbon and all bonds to single bonds, providing a generic framework.
    """
    smiles = "CC1=CC=C(C=C1)C(C)C(=O)O"  # Ibuprofen
    mol = Chem.MolFromSmiles(smiles)
    generator = MurckoScaffoldGenerator(make_generic=True)
    scaffold = generator.get_scaffold(mol)

    assert scaffold is not None, "Scaffold generation failed"
    # Check that all atoms in the generic scaffold are carbons (atomic number 6)
    for atom in scaffold.GetAtoms():
        assert atom.GetAtomicNum() == 6, "All atoms should be carbon in a generic scaffold"
    # Check that all bonds are single bonds
    for bond in scaffold.GetBonds():
        assert (
            bond.GetBondType() == Chem.rdchem.BondType.SINGLE
        ), "All bonds should be single in a generic scaffold"


def test_mol_to_scaffold_transformer_integration():
    """
    Tests the integration of MurckoScaffoldGenerator with MolToScaffoldTransformer
    within a scikit-learn Pipeline. This ensures the transformers correctly chain
    together, with the second transformer operating on the output of the first.
    """
    smiles_list = list(MURCKO_TEST_CASES.keys())

    # Define the pipeline
    pipeline = Pipeline([
        ('smiles_to_mol', SmilesToMolTransformer()),
        ('mol_to_scaffold', MolToScaffoldTransformer(
            scaffold_generator=MurckoScaffoldGenerator()
        ))
    ])

    # Transform the data through the pipeline
    scaffolds = pipeline.transform(smiles_list)

    assert len(scaffolds) == len(
        smiles_list
    ), "Pipeline should output one scaffold per input SMILES"

    # Check the output for a known case (Aspirin)
    aspirin_scaffold = scaffolds[0][0]
    assert isinstance(aspirin_scaffold, Chem.Mol), "Output should be an RDKit Mol object"
    assert Chem.MolToSmiles(aspirin_scaffold) == "c1ccccc1", "Incorrect scaffold for Aspirin"

    # Check the output for an acyclic case (Ethanol)
    # The scaffold should be an InvalidMol object as per the transformer's logic
    from scikit_mol.core import InvalidMol
    ethanol_scaffold = scaffolds[3][0]
    assert isinstance(ethanol_scaffold, InvalidMol), "Acyclic molecules should produce an InvalidMol object"


def test_scikit_learn_compatibility():
    """
    Verifies that the MurckoScaffoldGenerator is compatible with scikit-learn's
    `clone` function and can be used in a Pipeline. This is essential for
    hyperparameter tuning and other advanced scikit-learn workflows.
    """
    generator = MurckoScaffoldGenerator(include_chirality=True)
    # Test that the generator can be cloned
    cloned_generator = clone(generator)
    assert cloned_generator.include_chirality == generator.include_chirality
    assert cloned_generator is not generator, "Cloned object should be a new instance"

    # Test that the generator can be used in a scikit-learn Pipeline
    pipeline = Pipeline(
        [
            ("smiles_to_mol", SmilesToMolTransformer()),
            (
                "mol_to_scaffold",
                MolToScaffoldTransformer(scaffold_generator=generator),
            ),
        ]
    )
    # Cloning the pipeline should also work seamlessly
    cloned_pipeline = clone(pipeline)
    assert cloned_pipeline.steps[1][1].scaffold_generator.include_chirality

