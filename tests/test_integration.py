"""
tests.test_integration
======================
Tests for master dataset integration and relational integrity.
"""

from __future__ import annotations


def test_master_dataset_structure_and_counts(dataset):
    """Verify master dataset row count, enrollment sum, and unique counts."""
    master = dataset.master

    assert len(master) == 887, f"Expected 887 master rows, got {len(master)}"
    assert int(master["EnrollmentCount"].sum()) == 10000, (
        "Expected EnrollmentCount sum to be 10,000"
    )
    assert master["TeacherID"].nunique() == 60, "Expected 60 unique teachers in master"
    assert master["CourseID"].nunique() == 60, "Expected 60 unique courses in master"


def test_master_has_no_orphan_records(dataset):
    """Verify master contains no orphan teachers or courses."""
    master = dataset.master
    teachers = dataset.teachers
    courses = dataset.courses

    assert master["TeacherID"].isin(teachers["TeacherID"]).all(), "Found orphan TeacherID in master"
    assert master["CourseID"].isin(courses["CourseID"]).all(), "Found orphan CourseID in master"
    assert master["TeacherName"].notna().all(), "Found missing TeacherName in master"
    assert master["CourseName"].notna().all(), "Found missing CourseName in master"


def test_master_validation_flags_preserved(dataset):
    """Verify _has_teacher_rating and _has_course_rating flags match ratings presence."""
    master = dataset.master

    assert "_has_teacher_rating" in master.columns
    assert "_has_course_rating" in master.columns
    assert (master["_has_teacher_rating"] == master["TeacherRating"].notna()).all()
    assert (master["_has_course_rating"] == master["CourseRating"].notna()).all()
