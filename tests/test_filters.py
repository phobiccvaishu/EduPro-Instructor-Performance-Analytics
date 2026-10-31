"""
tests.test_filters
==================
Unit test suite verifying the filter model, slider bounds, semantic interaction,
empty-state behavior, serialization, and hashability.
"""

from __future__ import annotations

import pytest

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


SCOPE_FILTERS = [
    ("expertise", FilterState(expertise=("Machine Learning",))),
    ("course_category", FilterState(course_category=("Web Development",))),
    ("course_level", FilterState(course_level=("Advanced",))),
    ("instructor_gender", FilterState(instructor_gender=("Female",))),
    ("teacher_rating_range", FilterState(teacher_rating_range=(3.0, 4.97))),
    ("course_rating_range", FilterState(course_rating_range=(3.0, 4.94))),
    ("experience_range", FilterState(experience_range=(5.0, 24.0))),
]

SCOPE_TARGETS = [
    ("kpi_banner", "view.teacher_table & view.course_table"),
    ("tab1_rankings", "view.teacher_table"),
    ("tab2_experience", "view.teacher_table"),
    ("tab3_course_quality", "view.course_table"),
    ("tab4_domain_analysis", "view.master"),
    ("tab5_rating_relationships", "view.master"),
]

SCOPE_CASES = [
    (f_name, f_state, t_name, feed_obj, True)
    for (f_name, f_state) in SCOPE_FILTERS
    for (t_name, feed_obj) in SCOPE_TARGETS
]


@pytest.mark.parametrize(
    "filter_name,filter_state,target_name,feeding_obj,expected_changed",
    SCOPE_CASES,
)
def test_filter_scope_matrix_cell(
    dataset,
    filter_name,
    filter_state,
    target_name,
    feeding_obj,
    expected_changed,
):
    """
    Parametrized test for all 42 (filter, UI target) scope cells.
    Verifies that narrowing filters mutate their primary feeding data objects.
    """
    base_view = apply_filters(dataset, FilterState.default(dataset))
    filtered_view = apply_filters(dataset, filter_state)

    if target_name == "kpi_banner":
        changed = not filtered_view.teacher_table.equals(
            base_view.teacher_table
        ) or not filtered_view.course_table.equals(base_view.course_table)
    elif target_name in ("tab1_rankings", "tab2_experience"):
        changed = not filtered_view.teacher_table.equals(base_view.teacher_table)
    elif target_name == "tab3_course_quality":
        changed = not filtered_view.course_table.equals(base_view.course_table)
    elif target_name in ("tab4_domain_analysis", "tab5_rating_relationships"):
        changed = not filtered_view.master.equals(base_view.master)
    else:
        raise ValueError(f"Unknown target: {target_name}")

    assert changed == expected_changed, (
        f"Scope cell failed for ({filter_name}, {target_name}): "
        f"expected changed={expected_changed}, got changed={changed} on {feeding_obj}"
    )


def test_specific_plotted_aggregates_scope(dataset):
    """
    Item 23: Evaluates the exact plotted aggregate in each tab and documents honest no-effect cells:
    - Tab 1: Rankings total enrollments mutated on all 7 filters.
    - Tab 2: Intrinsic TeacherRating vs YearsOfExperience scatter is UNCHANGED under
      course_level and course_rating_range because all 60 instructors teach at least one advanced
      and high-rated course, leaving teacher demographics identical.
    - Tab 3: Heatmap matrix changes on all 7 filters.
    - Tab 4: Domain mean CourseRating changes on all 7 filters.
    - Tab 5: Master pair ratings change on all 7 filters.
    - Raw catalogs: view.courses unchanged under teacher filters; view.teachers unchanged under course filters.
    """
    base = apply_filters(dataset, FilterState.default(dataset))
    base_t2_scatter = (
        base.teacher_table[["YearsOfExperience", "TeacherRating"]]
        .sort_values(by="YearsOfExperience")
        .reset_index(drop=True)
    )

    for f_name, f_state in SCOPE_FILTERS:
        v = apply_filters(dataset, f_state)

        # Tab 1: Rankings enrollment aggregate
        assert not v.teacher_table["total_enrollments"].equals(
            base.teacher_table["total_enrollments"]
        )

        # Tab 2: Intrinsic Teacher demographics scatter
        v_t2_scatter = (
            v.teacher_table[["YearsOfExperience", "TeacherRating"]]
            .sort_values(by="YearsOfExperience")
            .reset_index(drop=True)
        )
        if f_name in ("course_level", "course_rating_range"):
            # Honest no-effect: all 60 instructors remain active
            assert v_t2_scatter.equals(base_t2_scatter), (
                f"Expected Tab 2 intrinsic scatter to be unchanged under {f_name}"
            )
        else:
            assert not v_t2_scatter.equals(base_t2_scatter), (
                f"Expected Tab 2 intrinsic scatter to change under {f_name}"
            )

        # Tab 3: Heatmap matrix
        v_hm = v.master.pivot_table(
            index="CourseCategory", columns="CourseLevel", values="CourseRating", aggfunc="mean"
        )
        base_hm = base.master.pivot_table(
            index="CourseCategory", columns="CourseLevel", values="CourseRating", aggfunc="mean"
        )
        assert not v_hm.equals(base_hm)

        # Tab 4: Domain mean CourseRating
        v_dm = v.master.groupby("CourseCategory")["CourseRating"].mean()
        base_dm = base.master.groupby("CourseCategory")["CourseRating"].mean()
        assert not v_dm.equals(base_dm)

        # Tab 5: Master pair ratings
        v_pairs = v.master[["TeacherRating", "CourseRating"]].reset_index(drop=True)
        base_pairs = base.master[["TeacherRating", "CourseRating"]].reset_index(drop=True)
        assert not v_pairs.equals(base_pairs)

        # Raw catalogs orthogonal invariance
        if f_name in (
            "expertise",
            "instructor_gender",
            "teacher_rating_range",
            "experience_range",
        ):
            assert v.courses.equals(base.courses)
        elif f_name in ("course_category", "course_level", "course_rating_range"):
            assert v.teachers.equals(base.teachers)
