"""
tests.conftest
==============
Pytest fixtures for EduPro analytics test suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from edupro.data.cleaning import clean_all
from edupro.data.loader import load_raw_sheets
from edupro.data.pipeline import Dataset, load_all


@pytest.fixture(scope="session")
def raw_sheets():
    """Load raw sheets once per test session."""
    return load_raw_sheets()


@pytest.fixture(scope="session")
def cleaned_tables_and_log(raw_sheets):
    """Clean tables and return (tables, cleaning_log)."""
    return clean_all(raw_sheets)


@pytest.fixture(scope="session")
def dataset() -> Dataset:
    """Load full Dataset via pipeline."""
    return load_all()


@pytest.fixture(scope="session")
def golden_baseline() -> dict[str, Any]:
    """Load golden baseline dictionary."""
    path = Path(__file__).parent / "golden_baseline.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)
