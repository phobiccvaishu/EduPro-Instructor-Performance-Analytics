"""
tests.test_imports
==================
Exhaustive package import tests: verifies that every Python module under the
``edupro`` package and the top-level ``app.py`` can be imported without raising
``ImportError`` or ``AttributeError``.

A single broken import will be caught here before it can reach the UI at
runtime.

Key assertions
--------------
* Every module discovered via ``pkgutil.walk_packages`` imports cleanly.
* ``app`` itself imports cleanly.
* ``format_slope`` is importable from its canonical location
  (``edupro.formatting``) **and** from ``edupro.analytics.kpis`` for backward
  compatibility.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import edupro


def _all_edupro_module_names() -> list[str]:
    """Return fully-qualified names for every module under the edupro package."""
    names: list[str] = []
    for module_info in pkgutil.walk_packages(
        path=edupro.__path__,
        prefix=edupro.__name__ + ".",
        onerror=lambda name: pytest.fail(f"pkgutil walk error on {name}"),
    ):
        names.append(module_info.name)
    return names


@pytest.mark.parametrize("module_name", _all_edupro_module_names())
def test_edupro_module_importable(module_name: str) -> None:
    """Every edupro sub-module must import without error."""
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        pytest.fail(f"ImportError importing {module_name}: {exc}")
    except AttributeError as exc:
        pytest.fail(f"AttributeError importing {module_name}: {exc}")


def test_app_importable() -> None:
    """Top-level app.py must import without error."""
    try:
        importlib.import_module("app")
    except ImportError as exc:
        pytest.fail(f"ImportError importing app: {exc}")
    except AttributeError as exc:
        pytest.fail(f"AttributeError importing app: {exc}")


def test_format_slope_importable_from_formatting() -> None:
    """format_slope must be importable from its canonical location."""
    try:
        from edupro.formatting import format_slope  # noqa: F401
    except ImportError as exc:
        pytest.fail(f"Cannot import format_slope from edupro.formatting: {exc}")


def test_format_slope_importable_from_kpis_backward_compat() -> None:
    """format_slope must remain importable from edupro.analytics.kpis (backward compat)."""
    try:
        from edupro.analytics.kpis import format_slope  # noqa: F401
    except ImportError as exc:
        pytest.fail(
            f"Backward-compat import of format_slope from edupro.analytics.kpis failed: {exc}"
        )


def test_format_slope_same_object() -> None:
    """The format_slope exported from kpis must be the exact same function as in formatting."""
    from edupro.analytics.kpis import format_slope as fs_kpis
    from edupro.formatting import format_slope as fs_fmt

    assert fs_kpis is fs_fmt, (
        "edupro.analytics.kpis.format_slope and edupro.formatting.format_slope "
        "must be the same object (re-export, not a copy)"
    )
