"""
tests.test_subset_robustness
============================
Evaluates robustness of all hypothesis tests (F1-F13) and KPI v2 computations
under constrained data slices:
  1. Single-expertise subset
  2. 5-teacher small sample subset
  3. Empty (0-row) subset
"""

from __future__ import annotations

from edupro.analytics.aggregation import course_table, teacher_table
from edupro.analytics.inference import (
    eval_f4a_experience_group_contrast,
    eval_f4b_experience_breakpoint,
    eval_f8a_teacher_rating_by_gender,
    eval_f8b_course_quality_by_gender,
    eval_f8c_gender_by_level_cells,
    run_all_inferences,
)
from edupro.analytics.kpis import KPIEngine
from edupro.config import AnalysisConfig


def _run_all_evaluators(teachers, courses, master, tt, ct, cfg):
    """Run all findings and KPIs and ensure zero unhandled exceptions."""
    kpi_engine = KPIEngine(master)
    kpis = kpi_engine.compute_all_kpis_v2(teachers, courses, teacher_df=tt, cfg=cfg)
    assert isinstance(kpis, dict)

    findings = run_all_inferences(teachers, courses, master, tt, ct, cfg=cfg)
    assert isinstance(findings, dict)
    assert len(findings) >= 13

    # Direct calls to split findings
    f4a = eval_f4a_experience_group_contrast(teachers, cfg)
    f4b = eval_f4b_experience_breakpoint(teachers, cfg)
    f8a = eval_f8a_teacher_rating_by_gender(teachers, cfg)
    f8b = eval_f8b_course_quality_by_gender(tt, cfg)
    f8c = eval_f8c_gender_by_level_cells(teachers, master, cfg)
    assert f4a.id == "F4a"
    assert f4b.id == "F4b"
    assert f8a.id == "F8a"
    assert f8b.id == "F8b"
    assert f8c.id == "F8c"


def test_single_expertise_subset_robustness(dataset):
    """Ensure analytical pipeline functions on single-expertise subset."""
    cfg = AnalysisConfig(fast_mode=True)
    target_exp = "Data Science"
    sub_teachers = dataset.teachers[dataset.teachers["Expertise"] == target_exp].copy()
    sub_master = dataset.master[dataset.master["TeacherID"].isin(sub_teachers["TeacherID"])].copy()
    tt = teacher_table(sub_teachers, sub_master)
    ct = course_table(dataset.courses, sub_master)

    _run_all_evaluators(sub_teachers, dataset.courses, sub_master, tt, ct, cfg)


def test_five_teacher_subset_robustness(dataset):
    """Ensure analytical pipeline functions on small 5-teacher subset."""
    cfg = AnalysisConfig(fast_mode=True)
    sub_teachers = dataset.teachers.iloc[:5].copy()
    sub_master = dataset.master[dataset.master["TeacherID"].isin(sub_teachers["TeacherID"])].copy()
    tt = teacher_table(sub_teachers, sub_master)
    ct = course_table(dataset.courses, sub_master)

    _run_all_evaluators(sub_teachers, dataset.courses, sub_master, tt, ct, cfg)


def test_empty_subset_robustness(dataset):
    """Ensure analytical pipeline functions gracefully without crashing on empty DataFrames."""
    cfg = AnalysisConfig(fast_mode=True)
    empty_teachers = dataset.teachers.iloc[0:0].copy()
    empty_courses = dataset.courses.iloc[0:0].copy()
    empty_master = dataset.master.iloc[0:0].copy()
    empty_tt = teacher_table(empty_teachers, empty_master)
    empty_ct = course_table(empty_courses, empty_master)

    _run_all_evaluators(empty_teachers, empty_courses, empty_master, empty_tt, empty_ct, cfg)
