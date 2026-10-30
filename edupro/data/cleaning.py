"""
edupro.data.cleaning
====================
Robust data cleaning pipeline preserving genuine nulls, preventing string 'nan'
pollution, tracking modifications in a structured CleaningLog, and cleaning all four sheets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from edupro.data.schema import (
    COURSES_COLS,
    REQUIRED_KEYS,
    TEACHERS_COLS,
    TRANSACTIONS_COLS,
    USERS_COLS,
)
from edupro.logging_setup import get_logger

logger = get_logger("data.cleaning")


@dataclass
class CleaningLog:
    """Detailed audit log recording row counts, dropped records, and string conversions."""

    initial_row_counts: dict[str, int] = field(default_factory=dict)
    final_row_counts: dict[str, int] = field(default_factory=dict)
    dropped_rows: dict[str, int] = field(default_factory=dict)
    null_strings_converted: dict[str, dict[str, int]] = field(default_factory=dict)
    rules_applied: list[str] = field(default_factory=list)

    def record_drop(self, sheet: str, count: int, reason: str) -> None:
        """Record rows dropped from a sheet."""
        self.dropped_rows[sheet] = self.dropped_rows.get(sheet, 0) + count
        if count > 0:
            self.rules_applied.append(f"Dropped {count} rows from '{sheet}' due to {reason}")

    def record_string_nulls(self, sheet: str, col: str, count: int) -> None:
        """Record empty strings or string 'nan' converted to NA."""
        if count > 0:
            if sheet not in self.null_strings_converted:
                self.null_strings_converted[sheet] = {}
            self.null_strings_converted[sheet][col] = count
            self.rules_applied.append(
                f"Normalized {count} empty/invalid strings to NA in '{sheet}.{col}'"
            )

    def summary(self) -> str:
        """Human-readable summary of cleaning operations."""
        lines = ["Cleaning Log Summary:"]
        for sheet, initial in self.initial_row_counts.items():
            final = self.final_row_counts.get(sheet, initial)
            dropped = self.dropped_rows.get(sheet, 0)
            lines.append(f"  {sheet}: {initial} raw -> {final} cleaned ({dropped} dropped)")
        if self.null_strings_converted:
            lines.append("  String normalization:")
            for sheet, cols in self.null_strings_converted.items():
                for col, cnt in cols.items():
                    lines.append(f"    {sheet}.{col}: {cnt} null-like strings converted to NA")
        return "\n".join(lines)


def _clean_text_series(series: pd.Series, sheet: str, col: str, log: CleaningLog) -> pd.Series:
    """
    Strip text, convert empty strings or literal 'nan'/'none' to pd.NA without
    converting existing genuine nulls to the string 'nan'.
    """
    # Track existing nulls
    was_na = series.isna()

    def _clean_str(val: Any) -> Any:
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        if s == "" or s.lower() in ("nan", "none", "null", "<na>"):
            return np.nan
        return s

    cleaned = series.apply(_clean_str)
    # Count how many non-null strings became NA
    new_na = cleaned.isna() & ~was_na
    count_converted = int(new_na.sum())
    if count_converted > 0:
        log.record_string_nulls(sheet, col, count_converted)

    return cleaned


def clean_teachers(df_raw: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Clean the Teachers sheet."""
    sheet = "Teachers"
    initial_len = len(df_raw)
    log.initial_row_counts[sheet] = initial_len

    df = df_raw[TEACHERS_COLS].copy()

    # Text columns
    for col in ("TeacherID", "TeacherName", "Gender", "Expertise"):
        df[col] = _clean_text_series(df[col], sheet, col, log)

    # Numeric columns
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["YearsOfExperience"] = pd.to_numeric(df["YearsOfExperience"], errors="coerce")
    df["TeacherRating"] = pd.to_numeric(df["TeacherRating"], errors="coerce")

    # Drop missing required keys
    req = REQUIRED_KEYS[sheet]
    before_drop = len(df)
    df.dropna(subset=req, inplace=True)
    df.reset_index(drop=True, inplace=True)
    dropped = before_drop - len(df)
    log.record_drop(sheet, dropped, f"missing required keys {req}")
    log.final_row_counts[sheet] = len(df)

    return df


def clean_courses(df_raw: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Clean the Courses sheet."""
    sheet = "Courses"
    initial_len = len(df_raw)
    log.initial_row_counts[sheet] = initial_len

    df = df_raw[COURSES_COLS].copy()

    # Text columns
    for col in ("CourseID", "CourseName", "CourseCategory", "CourseType", "CourseLevel"):
        df[col] = _clean_text_series(df[col], sheet, col, log)

    # Numeric columns - note: DO NOT fillna(0) on CoursePrice! Preserve NA for validation.
    df["CoursePrice"] = pd.to_numeric(df["CoursePrice"], errors="coerce")
    df["CourseDuration"] = pd.to_numeric(df["CourseDuration"], errors="coerce")
    df["CourseRating"] = pd.to_numeric(df["CourseRating"], errors="coerce")

    # Drop missing required keys
    req = REQUIRED_KEYS[sheet]
    before_drop = len(df)
    df.dropna(subset=req, inplace=True)
    df.reset_index(drop=True, inplace=True)
    dropped = before_drop - len(df)
    log.record_drop(sheet, dropped, f"missing required keys {req}")
    log.final_row_counts[sheet] = len(df)

    return df


def clean_transactions(df_raw: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Clean the Transactions sheet."""
    sheet = "Transactions"
    initial_len = len(df_raw)
    log.initial_row_counts[sheet] = initial_len

    df = df_raw[TRANSACTIONS_COLS].copy()

    # Text columns
    for col in ("TransactionID", "UserID", "CourseID", "TeacherID", "PaymentMethod"):
        df[col] = _clean_text_series(df[col], sheet, col, log)

    # Numeric / Date columns - note: DO NOT fillna(0) on Amount! Preserve NA for validation.
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")

    # Drop missing required keys
    req = REQUIRED_KEYS[sheet]
    before_drop = len(df)
    df.dropna(subset=req, inplace=True)
    df.reset_index(drop=True, inplace=True)
    dropped = before_drop - len(df)
    log.record_drop(sheet, dropped, f"missing required keys {req}")
    log.final_row_counts[sheet] = len(df)

    return df


def clean_users(df_raw: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Clean the Users sheet."""
    sheet = "Users"
    initial_len = len(df_raw)
    log.initial_row_counts[sheet] = initial_len

    df = df_raw[USERS_COLS].copy()

    # Text columns
    for col in ("UserID", "UserName", "Gender", "Email"):
        df[col] = _clean_text_series(df[col], sheet, col, log)

    # Numeric columns
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    # Drop missing required keys
    req = REQUIRED_KEYS[sheet]
    before_drop = len(df)
    df.dropna(subset=req, inplace=True)
    df.reset_index(drop=True, inplace=True)
    dropped = before_drop - len(df)
    log.record_drop(sheet, dropped, f"missing required keys {req}")
    log.final_row_counts[sheet] = len(df)

    return df


def clean_all(raw_sheets: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], CleaningLog]:
    """
    Clean all four sheets and return cleaned DataFrames along with an audit CleaningLog.

    Parameters
    ----------
    raw_sheets : dict[str, pd.DataFrame]
        Raw DataFrames keyed by sheet name.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], CleaningLog]
        Cleaned DataFrames and populated CleaningLog instance.
    """
    log = CleaningLog()
    cleaned = {
        "teachers": clean_teachers(raw_sheets["Teachers"], log),
        "courses": clean_courses(raw_sheets["Courses"], log),
        "transactions": clean_transactions(raw_sheets["Transactions"], log),
        "users": clean_users(raw_sheets["Users"], log),
    }
    logger.info("Data cleaning complete. %s", log.summary().replace("\n", " | "))
    return cleaned, log
