"""
tests.test_filters
==================
Unit test suite verifying the filter model, slider bounds, semantic interaction,
empty-state behavior, serialization, and hashability.
"""

from __future__ import annotations

import pytest

from edupro.analytics.engine import AnalyticsEngine
from edupro.config import AnalysisConfig
from edupro.filters import FilterBounds, FilterState, apply_filters, ceil_2d, floor_2d


def test_floor_ceil_2d_precision():
    """Verify floor_2d and ceil_2d do not suffer IEEE 754 precision issues on 2-decimal inputs."""
    assert floor_2d(1.13) == 1.13
    assert ceil_2d(4.94) == 4.94
    assert floor_2d(1.05) == 1.05
    assert ceil_2d(4.97) == 4.97
    assert floor_2d(1.1299999999999999) == 1.13
    assert ceil_2d(4.940000000000001) == 4.94


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

    # Exact equality checks for dataset bounds
    assert b.teacher_rating_min == 1.05
    assert b.teacher_rating_max == 4.97
    assert b.course_rating_min == 1.13
    assert b.course_rating_max == 4.94


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


@pytest.mark.parametrize(
    "filter_name,filter_state",
    [
        ("expertise", FilterState(expertise=("Machine Learning",))),
        ("course_category", FilterState(course_category=("Web Development",))),
        ("course_level", FilterState(course_level=("Advanced",))),
        ("instructor_gender", FilterState(instructor_gender=("Female",))),
        ("teacher_rating_range", FilterState(teacher_rating_range=(3.0, 4.97))),
        ("course_rating_range", FilterState(course_rating_range=(3.0, 4.94))),
        ("experience_range", FilterState(experience_range=(5.0, 24.0))),
    ],
)
def test_filter_scope_matrix_coverage(dataset, filter_name, filter_state):
    """Verify that every filter causes a measurable update across all 6 UI functional areas."""
    cfg = AnalysisConfig(fast_mode=True)
    base_view = apply_filters(dataset, FilterState.default(dataset))
    base_res = AnalyticsEngine(
        base_view.master,
        base_view.teachers,
        base_view.courses,
        base_view.transactions,
    ).run_full_analysis(cfg=cfg)

    view = apply_filters(dataset, filter_state)
    res = AnalyticsEngine(
        view.master,
        view.teachers,
        view.courses,
        view.transactions,
    ).run_full_analysis(cfg=cfg)

    # 1. Platform KPIs
    kpis_changed = (
        res["kpis_v2"]["avg_teacher_rating"].value
        != base_res["kpis_v2"]["avg_teacher_rating"].value
        or res["kpis_v2"]["avg_course_rating"].value
        != base_res["kpis_v2"]["avg_course_rating"].value
        or view.counts["n_enrollments"] != base_view.counts["n_enrollments"]
    )
    assert kpis_changed, f"KPIs did not change for {filter_name}"

    # 2. Tab 1: Instructor Rankings (view.teacher_table)
    tab1_changed = not view.teacher_table.equals(base_view.teacher_table)
    assert tab1_changed, f"Tab 1 (teacher_table) did not change for {filter_name}"

    # 3. Tab 2: Experience & Ratings (res['findings']['F2'] or teacher_table)
    tab2_changed = (
        not view.teacher_table.equals(base_view.teacher_table)
        or res["findings"]["F2"].statistic != base_res["findings"]["F2"].statistic
    )
    assert tab2_changed, f"Tab 2 (experience vs performance) did not change for {filter_name}"

    # 4. Tab 3: Course Quality Maps (view.course_table)
    tab3_changed = not view.course_table.equals(base_view.course_table)
    assert tab3_changed, f"Tab 3 (course_table) did not change for {filter_name}"

    # 5. Tab 4: Domain Analysis (res['findings']['F8a'] or master)
    tab4_changed = (
        not view.master.equals(base_view.master)
        or res["findings"]["F8a"].statistic != base_res["findings"]["F8a"].statistic
    )
    assert tab4_changed, f"Tab 4 (domain analysis) did not change for {filter_name}"

    # 6. Tab 5: Rating Relationships (view.master pair dataset)
    tab5_changed = not view.master.equals(base_view.master)
    assert tab5_changed, f"Tab 5 (master pair data) did not change for {filter_name}"
