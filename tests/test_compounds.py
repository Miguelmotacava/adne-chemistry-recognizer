"""Tests for data/compounds.py — schema, uniqueness, SMILES validity, filtering."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.compounds import COMPOUNDS, TAXONOMY, get_compounds, validate_schema


def test_compounds_not_empty():
    assert len(COMPOUNDS) >= 100, (
        f"Expected at least 100 compounds, got {len(COMPOUNDS)}"
    )


def test_schema_valid():
    errors = validate_schema()
    assert not errors, "Schema validation failed:\n  " + "\n  ".join(errors)


def test_unique_ids():
    ids = [c["id"] for c in COMPOUNDS]
    duplicates = [i for i in ids if ids.count(i) > 1]
    assert len(ids) == len(set(ids)), f"Duplicate IDs: {set(duplicates)}"


def test_smiles_validity():
    """Skip ionic compounds — their SMILES may not parse as neutral molecules."""
    try:
        from rdkit import Chem
    except ImportError:
        pytest.skip("RDKit not installed; cannot validate SMILES")

    failed = []
    for c in COMPOUNDS:
        if c.get("ionic", False):
            continue
        mol = Chem.MolFromSmiles(c["smiles"])
        if mol is None:
            failed.append((c["id"], c["smiles"]))
    assert not failed, (
        "Invalid SMILES for non-ionic compounds:\n  "
        + "\n  ".join(f"{cid}: {smi}" for cid, smi in failed)
    )


def test_taxonomy_structure():
    assert "inorganica" in TAXONOMY
    assert "organica" in TAXONOMY
    for cat_data in TAXONOMY.values():
        assert "display" in cat_data
        assert "subcategories" in cat_data
        for subcat_data in cat_data["subcategories"].values():
            assert "display" in subcat_data
            assert "ids" in subcat_data
            assert len(subcat_data["ids"]) > 0


def test_filter_by_category():
    inorg = get_compounds(category="inorganica")
    assert len(inorg) > 0
    assert all(c["category"] == "inorganica" for c in inorg)

    org = get_compounds(category="organica")
    assert len(org) > 0
    assert all(c["category"] == "organica" for c in org)


def test_filter_by_difficulty():
    basico = get_compounds(difficulty="basico")
    assert len(basico) > 0
    assert all(c["difficulty"] == "basico" for c in basico)


def test_filter_by_subcategories():
    res = get_compounds(category="organica",
                        subcategories=["alcanos", "alquenos"])
    assert len(res) > 0
    assert all(c["subcategory"] in {"alcanos", "alquenos"} for c in res)
