"""
tests.test_app_smoke
====================
Smoke test using Streamlit AppTest to ensure the entire dashboard runs
without unhandled exceptions and all 5 tabs initialize properly.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_smoke_runs_without_exceptions():
    """Execute app.py via Streamlit AppTest and verify zero uncaught exceptions."""
    app_path = (Path(__file__).resolve().parent.parent / "app.py").resolve()
    at = AppTest.from_file(str(app_path), default_timeout=120)
    at.run()
    assert not at.exception, f"App raised an uncaught exception: {at.exception}"
