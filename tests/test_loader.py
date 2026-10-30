"""
tests.test_loader
=================
Tests for single-pass Excel loader, schema validation, and missing item errors.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from edupro.data.loader import compute_sha256, load_raw_sheets
from edupro.data.schema import REQUIRED_SHEETS


def test_loader_loads_four_sheets_with_expected_counts(raw_sheets):
    """Verify all four required sheets load with exact raw counts."""
    for sheet in REQUIRED_SHEETS:
        assert sheet in raw_sheets, f"Sheet '{sheet}' missing from loaded raw sheets"

    assert len(raw_sheets["Users"]) == 3000
    assert len(raw_sheets["Teachers"]) == 60
    assert len(raw_sheets["Courses"]) == 60
    assert len(raw_sheets["Transactions"]) == 10000


def test_loader_file_not_found():
    """Verify clear FileNotFoundError when path does not exist."""
    fake_path = Path("non_existent_dir/no_file.xlsx")
    with pytest.raises(FileNotFoundError) as exc_info:
        load_raw_sheets(fake_path)
    assert "Data file not found" in str(exc_info.value)
    assert "EDUPRO_DATA_PATH" in str(exc_info.value)


def test_loader_missing_required_sheet(tmp_path):
    """Verify ValueError naming missing sheets if a required sheet is omitted."""
    dummy_file = tmp_path / "incomplete.xlsx"
    with pd.ExcelWriter(dummy_file, engine="openpyxl") as writer:
        pd.DataFrame({"TeacherID": ["T1"]}).to_excel(writer, sheet_name="Teachers", index=False)
        pd.DataFrame({"CourseID": ["C1"]}).to_excel(writer, sheet_name="Courses", index=False)
        # Missing 'Transactions' and 'Users'

    with pytest.raises(ValueError) as exc_info:
        load_raw_sheets(dummy_file)
    err = str(exc_info.value)
    assert "Missing required sheet" in err
    assert "Transactions" in err or "Users" in err


def test_loader_missing_required_column(tmp_path):
    """Verify ValueError naming missing columns if a required column is omitted."""
    dummy_file = tmp_path / "missing_col.xlsx"
    with pd.ExcelWriter(dummy_file, engine="openpyxl") as writer:
        pd.DataFrame({"TeacherID": ["T1"], "TeacherName": ["Alice"]}).to_excel(writer, sheet_name="Teachers", index=False)
        pd.DataFrame({"CourseID": ["C1"], "CourseRating": [4.0]}).to_excel(writer, sheet_name="Courses", index=False)
        pd.DataFrame({"TransactionID": ["TX1"]}).to_excel(writer, sheet_name="Transactions", index=False)
        pd.DataFrame({"UserID": ["U1"]}).to_excel(writer, sheet_name="Users", index=False)

    with pytest.raises(ValueError) as exc_info:
        load_raw_sheets(dummy_file)
    err = str(exc_info.value)
    assert "Missing required column" in err


def test_compute_sha256(tmp_path):
    """Verify compute_sha256 generates consistent hex digest."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("EduPro Test Content", encoding="utf-8")
    digest = compute_sha256(test_file)
    assert isinstance(digest, str)
    assert len(digest) == 64
