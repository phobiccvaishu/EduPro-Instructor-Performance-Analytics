"""
tests.test_filters
==================
Unit test suite verifying the filter model, slider bounds, semantic interaction,
empty-state behavior, serialization, and hashability.
"""

from __future__ import annotations

import pytest

from edupro.filters import FilterBounds, FilterState, apply_filters


def test_slider_bounds_cover_min_and_max_exactly(dataset):
    """Verify floor/ceil functions cover the extreme values (1.05 teacher, 4.94 course)."""
    tr_min = dataset.teachers["TeacherRating"].min()
    tr_max = dataset.teachers["TeacherRating"].max()
    cr_min = dataset.courses["CourseRating"].min()
    cr_max = dataset.courses["CourseRating"].max()

    b = FilterBounds.from_dataset(dataset)
    assert b.teacher_rating_min <= tr_min
    assert b.teacher_rating_max >= tr_max
    assert b.course_rating_min <= cr_min
    assert b.course_rating_max >= cr_max

    # Exact checks for previously truncated records
    assert b.teacher_rating_min <= 1.05
    assert b.course_rating_max >= 4.94


def test_default_filter_is_identity(dataset):
    """Assert default filter state returns exactly 60 teachers, 60 courses, 887 pairs, 10000 enrollments."""
    st = FilterState.default(dataset)
    assert st.is_default(dataset)

    view = apply_filters(dataset, st)
    assert view.counts["n_teachers"] == 60
    assert view.counts["n_courses"] == 60
    assert view.counts["n_pairs"] == 887
    assert view.counts["n_enrollments"] == 10000
    assert len(view.active_filters) == 0
    assert not view.is_empty


def test_single_teacher_filter_reduces_teachers(dataset):
    """Filtering on expertise reduces teachers, leaving courses intact until intersection."""
    st = FilterState(expertise=("Data Science",))
    view = apply_filters(dataset, st)

    exp_count = (dataset.teachers["Expertise"] == "Data Science").sum()
    assert view.counts["n_teachers"] == exp_count
    assert view.counts["n_teachers"] < 60
    assert view.counts["n_pairs"] < 887
    assert view.counts["n_enrollments"] < 10000
    assert "Expertise: Data Science" in view.active_filters


def test_single_course_filter_reduces_courses(dataset):
    """Filtering on course level reduces courses, leaving teachers intact until intersection."""
    st = FilterState(course_level=("Advanced",))
    view = apply_filters(dataset, st)

    lvl_count = (dataset.courses["CourseLevel"] == "Advanced").sum()
    assert view.counts["n_courses"] == lvl_count
    assert view.counts["n_courses"] < 60
    assert view.counts["n_pairs"] < 887
    assert "Course Level: Advanced" in view.active_filters


def test_teacher_course_interaction_and_no_enrollment_note(dataset):
    """When courses are filtered, teachers with 0 enrollments in scope are dropped from teacher_table."""
    st = FilterState(course_category=("Web Development",))
    view = apply_filters(dataset, st)

    # All pairs in master belong to Web Development courses
    cat_courses = dataset.courses.loc[
        dataset.courses["CourseCategory"] == "Web Development", "CourseID"
    ]
    assert set(view.master["CourseID"]).issubset(set(cat_courses))

    # Total enrollments in view match teacher_table and course_table
    tot_e = view.counts["n_enrollments"]
    assert view.teacher_table["total_enrollments"].sum() == tot_e
    assert view.course_table["total_enrollments"].sum() == tot_e
    assert view.teacher_table["enrollment_share"].sum() == pytest.approx(1.0, abs=1e-4)

    # Dropped teachers note present
    if view.counts["n_teachers_dropped"] > 0:
        assert any("no enrollments in the selected course scope" in n for n in view.notes)


def test_empty_filter_result_handled(dataset):
    """An impossible filter combination returns an empty view cleanly without errors."""
    st = FilterState(
        expertise=("Data Science",),
        teacher_rating_range=(1.0, 1.2),  # No Data Science teachers in this narrow low band
    )
    view = apply_filters(dataset, st)
    assert view.is_empty or len(view.teachers) == 0
    assert view.counts["n_enrollments"] == 0 or len(view.master) == 0


def test_filter_state_hashability_and_immutability():
    """Verify FilterState is frozen and hashable for caching."""
    st1 = FilterState(expertise=("Programming", "AI"))
    st2 = FilterState(expertise=("Programming", "AI"))
    assert hash(st1) == hash(st2)
    assert st1 == st2

    with pytest.raises(AttributeError):
        st1.expertise = ("Other",)


def test_filter_state_serialization_round_trip():
    """Verify to_dict and from_dict preserve all properties."""
    orig = FilterState(
        expertise=("Data Science",),
        course_category=("Business",),
        course_level=("Intermediate",),
        instructor_gender=("Female",),
        teacher_rating_range=(3.0, 4.5),
        course_rating_range=(2.5, 4.0),
        experience_range=(2.0, 15.0),
    )
    d = orig.to_dict()
    restored = FilterState.from_dict(d)
    assert orig == restored


def test_inverted_ranges_rejected():
    """Verify inverted ranges raise ValueError on construction."""
    with pytest.raises(ValueError):
        FilterState(teacher_rating_range=(4.5, 2.0))
    with pytest.raises(ValueError):
        FilterState(course_rating_range=(4.0, 1.0))
    with pytest.raises(ValueError):
        FilterState(experience_range=(20.0, 5.0))
