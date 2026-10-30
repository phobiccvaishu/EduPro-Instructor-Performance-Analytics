"""
edupro.data.pipeline
====================
End-to-end data pipeline orchestrating loading, cleaning, validation, and integration
into an immutable Dataset container.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from edupro.config import settings
from edupro.data.cleaning import CleaningLog, clean_all
from edupro.data.integration import build_master
from edupro.data.loader import compute_sha256, load_raw_sheets
from edupro.data.validation import ValidationReport, validate_all
from edupro.logging_setup import get_logger

logger = get_logger("data.pipeline")


@dataclass(frozen=True)
class Dataset:
    """Frozen container encapsulating the complete loaded and validated dataset."""

    teachers: pd.DataFrame
    courses: pd.DataFrame
    transactions: pd.DataFrame
    users: pd.DataFrame
    master: pd.DataFrame
    validation: ValidationReport
    cleaning_log: CleaningLog
    source_path: Path
    source_sha256: str
    loaded_at: datetime


def load_all(path: Path | str | None = None, strict: bool = True) -> Dataset:
    """
    Execute the full data pipeline from Excel to master dataset.

    Steps
    -----
    1. Read the workbook once using `load_raw_sheets`.
    2. Clean text, numeric, and date fields across all 4 sheets via `clean_all`.
    3. Validate schemas, completeness, referential integrity, ranges, and cardinality.
    4. If `strict` is True and validation has failures, raise a ValueError naming every failure.
    5. Construct the integrated master dataset via `build_master`.
    6. Return a frozen `Dataset` instance.

    Parameters
    ----------
    path : Path | str | None, optional
        Path to the Excel file. Defaults to `settings.data_path`.
    strict : bool, optional
        If True, raises an exception if any validation check has status FAIL. Default is True.

    Returns
    -------
    Dataset
        Immutable container with all tables, validation report, and metadata.

    Raises
    ------
    FileNotFoundError
        If data file cannot be found.
    ValueError
        If raw file schema is invalid or validation fails in strict mode.
    """
    resolved_path = Path(path) if path is not None else settings.data_path
    file_sha256 = compute_sha256(resolved_path)

    # 1. Load raw sheets (single pass)
    raw_sheets = load_raw_sheets(resolved_path)

    # 2. Clean all sheets
    cleaned_tables, cleaning_log = clean_all(raw_sheets)

    # 3. Validate
    validation_report = validate_all(cleaned_tables)

    # 4. Strict mode failure check
    if strict and validation_report.has_failures:
        failure_messages = [
            f"[{f.category}] {f.name}: {f.detail}" for f in validation_report.failures()
        ]
        error_msg = (
            f"Dataset validation failed with {len(failure_messages)} error(s) in strict mode:\n"
            + "\n".join(f"  - {msg}" for msg in failure_messages)
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # 5. Build master
    master = build_master(cleaned_tables)

    dataset = Dataset(
        teachers=cleaned_tables["teachers"],
        courses=cleaned_tables["courses"],
        transactions=cleaned_tables["transactions"],
        users=cleaned_tables["users"],
        master=master,
        validation=validation_report,
        cleaning_log=cleaning_log,
        source_path=resolved_path,
        source_sha256=file_sha256,
        loaded_at=datetime.now(timezone.utc),
    )
    logger.info("Pipeline load_all completed successfully from %s", resolved_path)
    return dataset
