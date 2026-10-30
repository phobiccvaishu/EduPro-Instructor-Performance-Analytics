"""
tests.test_cleaning
===================
Tests for data cleaning pipeline: no string 'nan' pollution, whitespace stripping,
audited dropping of missing keys, and preservation of legitimate 0.00 prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from edupro.data.cleaning import (
    CleaningLog,
    clean_courses,
    clean_teachers,
    clean_transactions,
)


def test_no_literal_nan_strings_in_cleaned_data(cleaned_tables_and_log):
    """Ensure no string columns contain the literal text 'nan', 'none', or 'null'."""
    tables, _ = cleaned_tables_and_log
    for sheet_name, df in tables.items():
        for col in df.select_dtypes(include=["object", "string"]).columns:
            str_vals = df[col].dropna().astype(str).str.lower().str.strip()
            assert not (str_vals == "nan").any(), (
                f"Found literal 'nan' string in {sheet_name}.{col}"
            )
            assert not (str_vals == "none").any(), (
                f"Found literal 'none' string in {sheet_name}.{col}"
            )
            assert not (str_vals == "null").any(), (
                f"Found literal 'null' string in {sheet_name}.{col}"
            )


def test_cleaning_preserves_zero_price_and_amount():
    """Verify free course price (0.00) is preserved as float 0.0 and not treated as missing."""
    log = CleaningLog()
    courses_raw = pd.DataFrame(
        {
            "CourseID": ["C_FREE", "C_PAID", "C_MISSING_PRICE"],
            "CourseName": ["Intro to Python", "Advanced ML", "Web Basics"],
            "CourseCategory": ["Programming", "AI", "Web"],
            "CourseType": ["Free", "Paid", "Paid"],
            "CourseLevel": ["Beginner", "Advanced", "Intermediate"],
            "CoursePrice": [0.0, 99.99, np.nan],
            "CourseDuration": [10.0, 25.0, 15.0],
            "CourseRating": [4.5, 4.8, 3.9],
        }
    )
    cleaned = clean_courses(courses_raw, log)
    assert len(cleaned) == 3
    assert cleaned.loc[cleaned["CourseID"] == "C_FREE", "CoursePrice"].iloc[0] == 0.0
    assert cleaned.loc[cleaned["CourseID"] == "C_PAID", "CoursePrice"].iloc[0] == 99.99
    assert pd.isna(cleaned.loc[cleaned["CourseID"] == "C_MISSING_PRICE", "CoursePrice"].iloc[0])


def test_cleaning_strips_whitespace_and_normalizes_empty_strings():
    """Verify leading/trailing whitespace is stripped and empty strings become NA."""
    log = CleaningLog()
    teachers_raw = pd.DataFrame(
        {
            "TeacherID": [" T1 ", "T2", "  "],
            "TeacherName": [" Alice Smith  ", "Bob Jones", "Ghost"],
            "Age": ["40 ", "35", "50"],
            "Gender": [" Female ", "Male", "Male"],
            "Expertise": [" AI ", "  Data Science  ", "Marketing"],
            "YearsOfExperience": ["10", " 5 ", "12"],
            "TeacherRating": [4.5, 3.8, 4.0],
        }
    )
    cleaned = clean_teachers(teachers_raw, log)

    # T1 had whitespace trimmed
    assert cleaned.loc[0, "TeacherID"] == "T1"
    assert cleaned.loc[0, "TeacherName"] == "Alice Smith"
    assert cleaned.loc[0, "Gender"] == "Female"
    assert cleaned.loc[0, "Expertise"] == "AI"
    assert cleaned.loc[0, "Age"] == 40.0
    assert cleaned.loc[0, "YearsOfExperience"] == 10.0

    # Row 3 had whitespace-only TeacherID "  ", which became NA, so it should be dropped
    assert "T1" in cleaned["TeacherID"].values
    assert "T2" in cleaned["TeacherID"].values
    assert len(cleaned) == 2
    assert log.dropped_rows["Teachers"] == 1


def test_cleaning_drops_rows_with_missing_required_keys():
    """Verify missing required keys are dropped and audited in CleaningLog."""
    log = CleaningLog()
    tx_raw = pd.DataFrame(
        {
            "TransactionID": ["TX1", "TX2", None, "TX4"],
            "UserID": ["U1", "U2", "U3", "U4"],
            "CourseID": ["C1", None, "C3", "C4"],
            "TransactionDate": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
            "Amount": [50.0, 50.0, 50.0, 50.0],
            "PaymentMethod": ["Credit Card", "UPI", "PayPal", "NetBanking"],
            "TeacherID": ["T1", "T2", "T3", None],
        }
    )
    cleaned = clean_transactions(tx_raw, log)
    # TX2 missing CourseID, TX3 missing TransactionID, TX4 missing TeacherID
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["TransactionID"] == "TX1"
    assert log.dropped_rows["Transactions"] == 3
