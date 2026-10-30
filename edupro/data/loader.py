"""
edupro.data.loader
==================
Single-pass Excel workbook loader with strict schema validation and SHA256 auditing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from edupro.config import settings
from edupro.data.schema import REQUIRED_SHEETS, SHEET_COLUMNS
from edupro.logging_setup import get_logger

logger = get_logger("data.loader")


def compute_sha256(filepath: Path | str) -> str:
    """Calculate the SHA256 hex digest of a file in 64KB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_raw_sheets(filepath: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """
    Read the Excel workbook once and return a dictionary of raw DataFrames.

    Validates presence of required sheets and required columns within each sheet.
    Strips leading and trailing whitespace from all column headers.

    Parameters
    ----------
    filepath : Path | str | None, optional
        Path to the Excel file. Defaults to `settings.data_path`.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping sheet names to raw DataFrames.

    Raises
    ------
    FileNotFoundError
        If the data file does not exist at the resolved path.
    ValueError
        If required sheets or required columns are missing from the workbook.
    """
    path = Path(filepath) if filepath is not None else settings.data_path

    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found at '{path}'. "
            "Please ensure the file exists or set the EDUPRO_DATA_PATH environment variable."
        )

    size_bytes = path.stat().st_size
    file_sha256 = compute_sha256(path)
    logger.info(
        "Loading workbook from %s (size: %d bytes, sha256: %s)",
        path,
        size_bytes,
        file_sha256,
    )

    raw_sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")

    # Check for missing sheets
    missing_sheets = [s for s in REQUIRED_SHEETS if s not in raw_sheets]
    if missing_sheets:
        raise ValueError(
            f"Missing required sheet(s) in workbook '{path}': {missing_sheets}. "
            f"Expected sheets: {list(REQUIRED_SHEETS)}."
        )

    # Normalize column names and validate required columns
    normalized_sheets: dict[str, pd.DataFrame] = {}
    for name in REQUIRED_SHEETS:
        df = raw_sheets[name].copy()
        df.columns = [str(c).strip() for c in df.columns]

        missing_cols = [c for c in SHEET_COLUMNS[name] if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required column(s) in sheet '{name}' of '{path}': {missing_cols}. "
                f"Expected columns: {SHEET_COLUMNS[name]}."
            )
        normalized_sheets[name] = df

    # Also keep any extra sheets present in the workbook
    for name, df in raw_sheets.items():
        if name not in normalized_sheets:
            df_copy = df.copy()
            df_copy.columns = [str(c).strip() for c in df_copy.columns]
            normalized_sheets[name] = df_copy

    return normalized_sheets
