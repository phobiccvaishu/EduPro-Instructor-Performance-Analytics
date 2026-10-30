"""
tests.test_characterization
===========================
Golden baseline characterization tests: verifies preservation of the core
data-model invariants and raw rating aggregates from golden_baseline.json.
"""

from __future__ import annotations

import pytest


def test_characterization_golden_baseline_match(dataset, golden_baseline):
    """Assert core data-model dimensions and raw aggregates match golden_baseline.json."""
    # 1. Sheet row counts
    assert len(dataset.users) == golden_baseline["sheet_row_counts"]["Users"]
    assert len(dataset.teachers) == golden_baseline["sheet_row_counts"]["Teachers"]
    assert len(dataset.courses) == golden_baseline["sheet_row_counts"]["Courses"]
    assert len(dataset.transactions) == golden_baseline["sheet_row_counts"]["Transactions"]

    # 2. Master transaction pair count (887 pairs)
    assert len(dataset.master) == golden_baseline["master_row_count"]
    assert len(dataset.master) == 887

    # 3. Total enrollment count (10,000 enrollments)
    assert int(dataset.master["EnrollmentCount"].sum()) == golden_baseline["enrollment_count_total"]
    assert int(dataset.master["EnrollmentCount"].sum()) == 10000

    # 4. Sheet-level average ratings
    assert float(dataset.teachers["TeacherRating"].mean()) == pytest.approx(
        golden_baseline["avg_teacher_rating"], abs=1e-4
    )
    assert float(dataset.courses["CourseRating"].mean()) == pytest.approx(
        golden_baseline["avg_course_rating"], abs=1e-4
    )
