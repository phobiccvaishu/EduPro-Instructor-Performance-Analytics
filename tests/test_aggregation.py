"""
tests.test_aggregation
======================
Unit tests for teacher_table and course_table aggregations.
"""

from __future__ import annotations

import pandas as pd
import pytest

from edupro.analytics.aggregation import course_table, teacher_table
from edupro.data.pipeline import load_all


class TestSyntheticAggregation:
    """Hand-computed verification of enrollment weighting."""

    def test_hand_computed_teacher_weighting(self) -> None:
        teachers = pd.DataFrame({
            "TeacherID": [1, 2],
            "TeacherName": ["Alice", "Bob"],
            "Age": [30, 40],
            "Gender": ["Female", "Male"],
            "Expertise": ["Python", "Data Science"],
            "YearsOfExperience": [5, 10],
            "TeacherRating": [4.5, 3.5],
        })

        # Alice teaches Course 101 (rating 4.0, 100 enr) and Course 102 (rating 5.0, 300 enr)
        # Expected weighted rating = (100*4.0 + 300*5.0) / 400 = 1900 / 400 = 4.75
        # Bob teaches Course 102 (rating 5.0, 200 enr)
        # Expected weighted rating = 5.0
        master = pd.DataFrame({
            "TeacherID": [1, 1, 2],
            "TeacherName": ["Alice", "Alice", "Bob"],
            "CourseID": [101, 102, 102],
            "CourseName": ["Py101", "DS102", "DS102"],
            "CourseCategory": ["Python", "Data Science", "Data Science"],
            "CourseRating": [4.0, 5.0, 5.0],
            "TeacherRating": [4.5, 4.5, 3.5],
            "EnrollmentCount": [100, 300, 200],
            "Expertise": ["Python", "Python", "Data Science"],
        })

        tt = teacher_table(teachers, master, attribution="enrollment_weighted")
        assert len(tt) == 2
        alice = tt[tt["TeacherID"] == 1].iloc[0]
        bob = tt[tt["TeacherID"] == 2].iloc[0]

        assert alice["total_enrollments"] == 400
        assert pytest.approx(alice["weighted_course_rating"], abs=1e-4) == 4.75
        # Alice own domain share: 100 out of 400 = 0.25
        assert pytest.approx(alice["own_domain_share"], abs=1e-4) == 0.25

        assert bob["total_enrollments"] == 200
        assert pytest.approx(bob["weighted_course_rating"], abs=1e-4) == 5.0
        assert pytest.approx(bob["own_domain_share"], abs=1e-4) == 1.0


class TestRealDataAggregation:
    """Invariants on real platform data."""

    def test_entity_table_invariants(self) -> None:
        ds = load_all()
        tt = teacher_table(ds.teachers, ds.master)
        ct = course_table(ds.courses, ds.master)

        # Invariants: 60 teachers, 60 courses
        assert len(tt) == 60
        assert len(ct) == 60

        # Total enrollment conservation: 10,000 enrollments
        assert tt["total_enrollments"].sum() == 10000
        assert ct["total_enrollments"].sum() == 10000

        # Enrollment share sums to 1.0
        assert pytest.approx(tt["enrollment_share"].sum(), abs=1e-4) == 1.0

        # Teacher tiers match config
        assert set(tt["InstructorTier"].unique()).issubset({"High (≥4.0)", "Mid (3.0–3.9)", "Low (<3.0)"})

    def test_sensitivity_min_enrollment_filter(self) -> None:
        ds = load_all()
        tt_filtered = teacher_table(ds.teachers, ds.master, min_pair_enrollment=3)
        assert len(tt_filtered) == 60
        # Dropping pairs with < 3 enrollments means total enrollments < 10000
        assert tt_filtered["total_enrollments"].sum() < 10000
