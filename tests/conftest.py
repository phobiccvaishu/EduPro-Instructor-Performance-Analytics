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


@pytest.fixture(autouse=True, scope="session")
def patch_windows_asyncio_for_streamlit_apptest():
    """
    Opt-in mock for restricted Windows sandbox environments where socket.socketpair() hangs.
    Activated only when EDUPRO_TEST_SANDBOX=1 is explicitly set in the environment.
    """
    import os
    import sys

    if sys.platform == "win32" and os.environ.get("EDUPRO_TEST_SANDBOX") == "1":
        import asyncio
        from unittest.mock import MagicMock

        orig_new_event_loop = asyncio.new_event_loop

        def safe_new_event_loop():
            m = MagicMock(spec=asyncio.AbstractEventLoop)
            m.is_closed.return_value = False
            return m

        asyncio.new_event_loop = safe_new_event_loop
        yield
        asyncio.new_event_loop = orig_new_event_loop
    else:
        yield
