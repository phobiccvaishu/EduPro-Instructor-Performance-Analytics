"""
tests.test_validation
=====================
Tests for validation report: passes on clean real data, emits INFO and WARN,
and correctly detects FAIL/WARN on corrupted synthetic datasets.
"""

from __future__ import annotations

import pandas as pd

from edupro.data.validation import CheckStatus, validate_all


def test_validation_passes_on_real_data(cleaned_tables_and_log):
    """Verify validation on real dataset produces no FAIL, with expected WARN and INFO."""
    tables, _ = cleaned_tables_and_log
    report = validate_all(tables)

    assert not report.has_failures, (
        f"Unexpected validation failures: {[f.name for f in report.failures()]}"
    )
    assert report.has_warnings, "Expected at least one warning (e.g. small groups)"

    # Check for expected INFO check results
    categories = [r.category for r in report.results]
    assert "Relationship" in categories
    assert "Expertise Alignment" in categories

    # Check the small groups warning
    warn_names = [w.name for w in report.warnings()]
    assert any("Small Groups" in name for name in warn_names)


def test_validation_fails_on_orphan_teacher_id(cleaned_tables_and_log):
    """Corrupt transactions with an orphan TeacherID and assert FAIL."""
    tables, _ = cleaned_tables_and_log
    corrupted = {k: v.copy() for k, v in tables.items()}
    # Introduce orphan TeacherID
    corrupted["transactions"].loc[0, "TeacherID"] = "NON_EXISTENT_TEACHER_999"

    report = validate_all(corrupted)
    assert report.has_failures
    orphan_checks = [f for f in report.failures() if "Transactions.TeacherID" in f.name]
    assert len(orphan_checks) == 1
    assert orphan_checks[0].status == CheckStatus.FAIL


def test_validation_fails_on_duplicate_id(cleaned_tables_and_log):
    """Corrupt courses with duplicate CourseID and assert FAIL."""
    tables, _ = cleaned_tables_and_log
    corrupted = {k: v.copy() for k, v in tables.items()}
    # Duplicate first course
    duplicate_row = corrupted["courses"].iloc[[0]].copy()
    corrupted["courses"] = pd.concat([corrupted["courses"], duplicate_row], ignore_index=True)

    report = validate_all(corrupted)
    assert report.has_failures
    dup_checks = [f for f in report.failures() if "Courses.CourseID" in f.name]
    assert len(dup_checks) == 1
    assert dup_checks[0].status == CheckStatus.FAIL


def test_validation_fails_on_rating_out_of_bounds(cleaned_tables_and_log):
    """Corrupt TeacherRating with 7.2 and assert FAIL."""
    tables, _ = cleaned_tables_and_log
    corrupted = {k: v.copy() for k, v in tables.items()}
    corrupted["teachers"].loc[0, "TeacherRating"] = 7.2

    report = validate_all(corrupted)
    assert report.has_failures
    rating_checks = [f for f in report.failures() if "TeacherRating in [1.0, 5.0]" in f.name]
    assert len(rating_checks) == 1
    assert rating_checks[0].status == CheckStatus.FAIL


def test_validation_fails_on_negative_price(cleaned_tables_and_log):
    """Corrupt CoursePrice with -19.99 and assert FAIL."""
    tables, _ = cleaned_tables_and_log
    corrupted = {k: v.copy() for k, v in tables.items()}
    corrupted["courses"].loc[0, "CoursePrice"] = -19.99

    report = validate_all(corrupted)
    assert report.has_failures
    price_checks = [f for f in report.failures() if "Courses.CoursePrice >= 0" in f.name]
    assert len(price_checks) == 1
    assert price_checks[0].status == CheckStatus.FAIL


def test_validation_fails_on_future_transaction_date(cleaned_tables_and_log):
    """Corrupt TransactionDate with 2099-01-01 and assert FAIL."""
    tables, _ = cleaned_tables_and_log
    corrupted = {k: v.copy() for k, v in tables.items()}
    corrupted["transactions"].loc[0, "TransactionDate"] = pd.Timestamp("2099-01-01")

    report = validate_all(corrupted)
    assert report.has_failures
    date_checks = [f for f in report.failures() if "TransactionDate" in f.name]
    assert len(date_checks) == 1
    assert date_checks[0].status == CheckStatus.FAIL
