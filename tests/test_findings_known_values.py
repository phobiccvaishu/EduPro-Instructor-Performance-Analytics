"""
tests.test_findings_known_values
================================
Regression verification tests asserting exact calibrated values for findings
F1, F4a, F4b, F8a, F8b, F8c, F12, and F13.
"""

from __future__ import annotations

import pytest

from edupro.analytics.aggregation import teacher_table
from edupro.analytics.inference import (
    eval_f1_instructor_distribution,
    eval_f4a_experience_group_contrast,
    eval_f4b_experience_breakpoint,
    eval_f8a_teacher_rating_by_gender,
    eval_f8b_course_quality_by_gender,
    eval_f8c_gender_by_level_cells,
    eval_f12_expertise_vs_teacher_rating,
    eval_f13_expertise_vs_course_quality_and_ml_gap,
)
from edupro.config import default_analysis_config


def test_f1_known_distribution_values(dataset):
    """F1: Deduplicated n=60 TeacherRating five-number summary, Shapiro-Wilk test, and tier counts."""
    tr = dataset.teachers["TeacherRating"].dropna().to_numpy(dtype=float)
    assert len(tr) == 60
    assert float(tr.sum()) == pytest.approx(187.5, abs=1e-5)
    assert float(tr.mean()) == pytest.approx(3.125, abs=1e-5)
    assert int((tr >= 4.0).sum()) == 11
    assert int(((tr >= 3.0) & (tr < 4.0)).sum()) == 28
    assert int((tr < 3.0).sum()) == 21
    assert int((tr < 2.0).sum()) == 9

    # Item 2: Tier sums and means (<3.0, 3.0-3.99, >=4.0)
    low_tr = tr[tr < 3.0]
    mid_tr = tr[(tr >= 3.0) & (tr < 4.0)]
    high_tr = tr[tr >= 4.0]
    assert len(low_tr) == 21
    assert float(low_tr.sum()) == pytest.approx(43.47, abs=1e-5)
    assert float(low_tr.mean()) == pytest.approx(2.070, abs=1e-3)

    assert len(mid_tr) == 28
    assert float(mid_tr.sum()) == pytest.approx(96.09, abs=1e-5)
    assert float(mid_tr.mean()) == pytest.approx(3.4318, abs=1e-3)

    assert len(high_tr) == 11
    assert float(high_tr.sum()) == pytest.approx(47.94, abs=1e-5)
    assert float(high_tr.mean()) == pytest.approx(4.3582, abs=1e-3)

    f1 = eval_f1_instructor_distribution(dataset.teachers, default_analysis_config)
    assert f1.n == 60
    assert f1.statistic == pytest.approx(3.125, abs=1e-3)
    assert f1.p_value == pytest.approx(0.0583, abs=1e-3)

    five_num = f1.extra["five_number_summary"]
    assert five_num["min"] == pytest.approx(1.050, abs=1e-3)
    assert five_num["q1"] == pytest.approx(2.488, abs=1e-3)
    assert five_num["median"] == pytest.approx(3.275, abs=1e-3)
    assert five_num["q3"] == pytest.approx(3.828, abs=1e-3)
    assert five_num["max"] == pytest.approx(4.970, abs=1e-3)

    assert f1.extra["iqr"] == pytest.approx(1.340, abs=1e-3)
    assert f1.extra["skewness"] == pytest.approx(-0.485, abs=1e-3)
    assert f1.extra["shapiro_statistic"] == pytest.approx(0.962, abs=1e-3)
    assert f1.extra["shapiro_p"] == pytest.approx(0.0583, abs=1e-3)


def test_f4_known_contrast_and_breakpoint(dataset):
    """F4a (0-5 vs 6+ contrast) and F4b (breakpoint search candidate)."""
    f4a = eval_f4a_experience_group_contrast(dataset.teachers, default_analysis_config)
    assert f4a.n == 60
    assert f4a.statistic == pytest.approx(1.361, abs=1e-2)
    assert f4a.verdict == "supported"
    assert f4a.p_value < 0.001

    f4b = eval_f4b_experience_breakpoint(dataset.teachers, default_analysis_config)
    assert f4b.n == 60
    assert f4b.statistic == pytest.approx(8.0, abs=1e-1)
    assert f4b.p_value == pytest.approx(0.128, abs=0.03)
    assert f4b.verdict == "not supported"


def test_f8_known_gender_family(dataset):
    """F8a (TeacherRating), F8b (CourseQuality), F8c (Gender x Level table)."""
    tt = teacher_table(dataset.teachers, dataset.master)
    f8a = eval_f8a_teacher_rating_by_gender(dataset.teachers, default_analysis_config)
    assert f8a.extra["source_column"] == "TeacherRating"
    assert f8a.extra["mean_female"] == pytest.approx(3.231, abs=1e-3)
    assert f8a.extra["mean_male"] == pytest.approx(3.038, abs=1e-3)
    assert f8a.statistic == pytest.approx(551.0, abs=1.0)
    assert f8a.extra["raw_p_value"] == pytest.approx(0.119, abs=0.01)
    assert f8a.verdict == "not supported"

    f8b = eval_f8b_course_quality_by_gender(tt, default_analysis_config)
    assert f8b.extra["source_column"] == "weighted_course_rating"
    assert f8b.extra["mean_female"] == pytest.approx(3.053, abs=1e-3)
    assert f8b.extra["mean_male"] == pytest.approx(3.247, abs=1e-3)
    assert f8b.statistic == pytest.approx(254.0, abs=1.0)
    assert f8b.extra["raw_p_value"] == pytest.approx(0.0045, abs=1e-3)
    assert f8b.p_value == pytest.approx(0.0091, abs=1e-3)
    assert f8b.verdict == "supported"
    assert "due to" not in f8b.notes.lower()

    # Leave-top-3 check
    sens3 = f8b.extra["leave_top_3_sensitivity"]
    assert sens3["status"] == "ok"
    assert sens3["p_value"] < 0.05
    assert sens3["effect"] == "moderate"

    f8c = eval_f8c_gender_by_level_cells(dataset.teachers, dataset.master, default_analysis_config)
    assert f8c.verdict == "descriptive"
    assert len(f8c.extra["cell_table"]) == 6


def test_f12_known_expertise_thresholds(dataset):
    """F12: Headline min_group_size=3 and secondary min_group_size=2."""
    f12 = eval_f12_expertise_vs_teacher_rating(dataset.teachers, default_analysis_config)
    assert f12.n == 56
    assert f12.statistic == pytest.approx(15.196, abs=1e-2)
    assert f12.p_value == pytest.approx(0.0554, abs=1e-3)
    assert f12.verdict == "not supported"

    sec = f12.extra["secondary_min_group_size_2"]
    assert sec["n"] == 58
    assert sec["statistic"] == pytest.approx(15.670, abs=1e-2)
    assert sec["p_value"] == pytest.approx(0.0741, abs=1e-3)


def test_f13_known_ml_curriculum_gap(dataset):
    """F13: Single method cluster bootstrap for B - A with catalog benchmark A held fixed."""
    tt = teacher_table(dataset.teachers, dataset.master)
    f13 = eval_f13_expertise_vs_course_quality_and_ml_gap(
        tt, dataset.courses, dataset.master, default_analysis_config
    )
    assert f13.statistic == pytest.approx(0.397, abs=1e-2)
    assert f13.extra["ml_quality_gap"]["A_ml_category_catalog_cr"] == pytest.approx(2.658, abs=1e-2)
    assert f13.extra["ml_quality_gap"]["B_ml_teachers_weighted_cr"] == pytest.approx(
        3.055, abs=1e-2
    )
    assert f13.extra["ml_quality_gap"]["bootstrap_mean"] == pytest.approx(0.289, abs=1e-2)
    assert f13.ci_low == pytest.approx(-0.066, abs=1e-2)
    assert f13.ci_high == pytest.approx(0.418, abs=1e-2)
    assert f13.p_value == pytest.approx(0.163, abs=0.02)
    assert f13.verdict == "not supported"
    assert f13.evidence == "None"
    assert f13.extra["ml_instructors_teacher_rating_mean"] == pytest.approx(4.234, abs=1e-2)
    assert f13.extra["ml_quality_gap"]["platform_alignment_share"] == pytest.approx(
        0.4111, abs=1e-3
    )
    # Item 4: Own-domain shares (a) platform-wide, (b) per-instructor enrollments, (c) per-instructor courses
    assert f13.extra["own_domain_enrollment_share_mean"] == pytest.approx(0.7396, abs=1e-3)
    assert f13.extra["own_domain_enrollment_share_median"] == pytest.approx(0.7722, abs=1e-3)
    assert f13.extra["own_domain_course_share_mean"] == pytest.approx(0.3932, abs=1e-3)
    assert f13.extra["own_domain_course_share_median"] == pytest.approx(0.3333, abs=1e-3)
    loo = f13.extra["ml_quality_gap"]["leave_one_cluster_out"]
    assert len(loo) == 5
    assert "TC00042" in loo
    assert loo["TC00042"]["B_minus_A_loo"] == pytest.approx(0.0836, abs=1e-3)
