"""
tests.test_filtered_analytics
=============================
Verifies analytical calculations, entity aggregations, and KPI integrity
under various filtered data slices.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from edupro.analytics.aggregation import course_table, teacher_table
from edupro.analytics.engine import AnalyticsEngine
from edupro.config import AnalysisConfig
from edupro.filters import FilterState, apply_filters


def test_entity_table_enrollments_sum_to_view_total(dataset):
    """Enrollment counts in recomputed teacher_table and course_table must match master enrollments."""
    st = FilterState(
        expertise=("Data Science", "Machine Learning"),
        course_level=("Advanced", "Intermediate"),
    )
    view = apply_filters(dataset, st)
    assert not view.is_empty

    tot_enr = view.counts["n_enrollments"]
    assert view.teacher_table["total_enrollments"].sum() == tot_enr
    assert view.course_table["total_enrollments"].sum() == tot_enr
    assert view.teacher_table["enrollment_share"].sum() == pytest.approx(1.0, abs=1e-4)
    assert view.course_table["enrollment_share"].sum() == pytest.approx(1.0, abs=1e-4)


def test_weighted_ratings_recomputed_correctly_on_synthetic_subset():
    """Verify hand-built synthetic pairs calculate the exact weighted rating within subset."""
    synth_teachers = pd.DataFrame(
        [
            {
                "TeacherID": "T1",
                "TeacherName": "Alice",
                "Age": 40,
                "Gender": "Female",
                "Expertise": "AI",
                "YearsOfExperience": 10,
                "TeacherRating": 4.5,
            },
            {
                "TeacherID": "T2",
                "TeacherName": "Bob",
                "Age": 35,
                "Gender": "Male",
                "Expertise": "AI",
                "YearsOfExperience": 5,
                "TeacherRating": 3.8,
            },
        ]
    )
    synth_courses = pd.DataFrame(
        [
            {
                "CourseID": "C1",
                "CourseName": "Intro AI",
                "CourseCategory": "AI",
                "CourseType": "Theory",
                "CourseLevel": "Beginner",
                "CoursePrice": 100,
                "CourseDuration": 20,
                "CourseRating": 4.0,
            },
            {
                "CourseID": "C2",
                "CourseName": "Deep AI",
                "CourseCategory": "AI",
                "CourseType": "Practical",
                "CourseLevel": "Advanced",
                "CoursePrice": 200,
                "CourseDuration": 40,
                "CourseRating": 4.6,
            },
        ]
    )
    # T1 teaches C1 (100 enr) and C2 (300 enr) -> weighted course rating = (100*4.0 + 300*4.6) / 400 = (400 + 1380)/400 = 1780/400 = 4.45
    # T2 teaches C1 (200 enr) -> weighted course rating = 4.0
    synth_master = pd.DataFrame(
        [
            {
                "TeacherID": "T1",
                "CourseID": "C1",
                "EnrollmentCount": 100,
                "TeacherRating": 4.5,
                "CourseRating": 4.0,
                "Expertise": "AI",
                "CourseCategory": "AI",
                "CourseLevel": "Beginner",
                "YearsOfExperience": 10,
                "TeacherName": "Alice",
                "CourseName": "Intro AI",
                "_has_teacher_rating": True,
                "_has_course_rating": True,
            },
            {
                "TeacherID": "T1",
                "CourseID": "C2",
                "EnrollmentCount": 300,
                "TeacherRating": 4.5,
                "CourseRating": 4.6,
                "Expertise": "AI",
                "CourseCategory": "AI",
                "CourseLevel": "Advanced",
                "YearsOfExperience": 10,
                "TeacherName": "Alice",
                "CourseName": "Deep AI",
                "_has_teacher_rating": True,
                "_has_course_rating": True,
            },
            {
                "TeacherID": "T2",
                "CourseID": "C1",
                "EnrollmentCount": 200,
                "TeacherRating": 3.8,
                "CourseRating": 4.0,
                "Expertise": "AI",
                "CourseCategory": "AI",
                "CourseLevel": "Beginner",
                "YearsOfExperience": 5,
                "TeacherName": "Bob",
                "CourseName": "Intro AI",
                "_has_teacher_rating": True,
                "_has_course_rating": True,
            },
        ]
    )

    tt = teacher_table(synth_teachers, synth_master)
    t1_row = tt[tt["TeacherID"] == "T1"].iloc[0]
    t2_row = tt[tt["TeacherID"] == "T2"].iloc[0]

    assert t1_row["total_enrollments"] == 400
    assert t1_row["weighted_course_rating"] == pytest.approx(4.45, abs=1e-4)
    assert t2_row["total_enrollments"] == 200
    assert t2_row["weighted_course_rating"] == pytest.approx(4.00, abs=1e-4)

    # C1 taught by T1 (100 enr, rating 4.5) and T2 (200 enr, rating 3.8)
    # -> weighted teacher rating = (100*4.5 + 200*3.8)/300 = (450 + 760)/300 = 1210/300 = 4.0333
    ct = course_table(synth_courses, synth_master)
    c1_row = ct[ct["CourseID"] == "C1"].iloc[0]
    assert c1_row["total_enrollments"] == 300
    assert c1_row["weighted_teacher_rating"] == pytest.approx(1210.0 / 300.0, abs=1e-4)


@pytest.mark.parametrize(
    "scenario,filter_state",
    [
        ("single_expertise", FilterState(expertise=("Data Science",))),
        ("single_category", FilterState(course_category=("Programming",))),
        ("five_teacher_subset", FilterState(teacher_rating_range=(4.2, 4.97))),
        (
            "narrow_rating_range",
            FilterState(teacher_rating_range=(3.0, 3.4), course_rating_range=(3.0, 3.4)),
        ),
    ],
)
def test_no_nan_in_kpi_tiles_across_filtered_subsets(dataset, scenario, filter_state):
    """Ensure no NaN in legacy or v2 KPI tiles across various filter slices."""
    cfg = AnalysisConfig(fast_mode=True)
    view = apply_filters(dataset, filter_state)

    if view.is_empty:
        pytest.skip(f"Scenario {scenario} produced empty view")

    engine = AnalyticsEngine(
        view.master,
        view.teachers,
        view.courses,
        view.transactions,
    )
    # Legacy KPIs
    legacy_kpis = engine.kpi.compute_all_kpis()
    assert not math.isnan(legacy_kpis["avg_teacher_rating"])
    assert not math.isnan(legacy_kpis["avg_course_rating"])
    assert not math.isnan(legacy_kpis["rating_consistency_index"])
    assert not math.isnan(legacy_kpis["experience_impact"]["impact_score"])
    assert not math.isnan(legacy_kpis["enrollment_influence"]["enrollment_influence_ratio"])

    # KPI v2
    kpis_v2 = engine.kpi.compute_all_kpis_v2(
        view.teachers,
        view.courses,
        teacher_df=view.teacher_table,
        cfg=cfg,
    )
    for name, kpi_res in kpis_v2.items():
        assert not math.isnan(kpi_res.value), f"NaN value in KPI v2: {name} under {scenario}"
        assert kpi_res.evidence in ("None", "Weak", "Moderate", "Strong", "Insufficient data")
