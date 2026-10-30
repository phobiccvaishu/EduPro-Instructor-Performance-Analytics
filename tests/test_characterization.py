"""
tests.test_characterization
===========================
Golden baseline characterization tests: verifies byte-for-byte or 1e-4 tolerance
reproduction of all pre-refactoring analytics outputs and KPIs.
"""

from __future__ import annotations

import math

import pytest

from edupro.analytics.engine import AnalyticsEngine


def test_characterization_golden_baseline_match(dataset, golden_baseline):
    """Assert all golden metrics match the pre-refactoring baseline within 1e-4 tolerance."""
    engine = AnalyticsEngine(
        dataset.master,
        dataset.teachers,
        dataset.courses,
        dataset.transactions,
    )
    results = engine.run_full_analysis()
    kpis = results["kpis"]

    # 1. Sheet counts & master counts
    assert len(dataset.users) == golden_baseline["sheet_row_counts"]["Users"]
    assert len(dataset.teachers) == golden_baseline["sheet_row_counts"]["Teachers"]
    assert len(dataset.courses) == golden_baseline["sheet_row_counts"]["Courses"]
    assert len(dataset.transactions) == golden_baseline["sheet_row_counts"]["Transactions"]
    assert len(dataset.master) == golden_baseline["master_row_count"]
    assert int(dataset.master["EnrollmentCount"].sum()) == golden_baseline["enrollment_count_total"]

    # 2. Platform KPIs
    assert kpis["avg_teacher_rating"] == pytest.approx(
        golden_baseline["avg_teacher_rating"], abs=1e-4
    )
    assert kpis["avg_course_rating"] == pytest.approx(
        golden_baseline["avg_course_rating"], abs=1e-4
    )
    assert kpis["rating_consistency_index"] == pytest.approx(
        golden_baseline["rating_consistency_index"], abs=1e-4
    )

    # 3. Experience impact
    exp_imp = kpis["experience_impact"]
    gold_exp = golden_baseline["experience_impact"]
    for key in (
        "r_exp_teacher",
        "p_exp_teacher",
        "slope_exp_teacher",
        "r_exp_course",
        "p_exp_course",
        "slope_exp_course",
        "impact_score",
    ):
        assert exp_imp[key] == pytest.approx(gold_exp[key], abs=1e-4), f"Mismatch in {key}"

    # 4. Enrollment influence
    enr_inf = kpis["enrollment_influence"]
    gold_enr = golden_baseline["enrollment_influence"]
    for key in (
        "avg_enrollment_high",
        "avg_enrollment_low",
        "avg_enrollment_all",
        "enrollment_influence_ratio",
    ):
        assert enr_inf[key] == pytest.approx(gold_enr[key], abs=1e-4), f"Mismatch in {key}"

    # 5. Tier counts
    impact_tier_summary = results["instructor_impact"]["tier_summary"]
    computed_tier_counts = dict(
        zip(
            impact_tier_summary["InstructorTier"],
            impact_tier_summary["instructor_count"],
            strict=True,
        )
    )
    assert computed_tier_counts == golden_baseline["tier_counts"]

    # 6. Stage-3 experience vs performance
    evp = results["experience_vs_performance"]
    gold_evp = golden_baseline["stage_3_experience_vs_performance"]
    for key in (
        "pearson_r_teacher",
        "pearson_p_teacher",
        "spearman_r_teacher",
        "spearman_p_teacher",
        "pearson_r_course",
        "pearson_p_course",
    ):
        assert evp[key] == pytest.approx(
            gold_exp[key] if key in gold_exp else gold_evp[key], abs=1e-4
        ), f"Mismatch in {key}"

    # 7. Teacher course correlation
    tc = results["teacher_course_corr"]
    gold_tc = golden_baseline["teacher_course_corr"]
    for key in ("pearson_r", "pearson_p", "spearman_r", "spearman_p", "slope", "intercept", "n"):
        assert tc[key] == pytest.approx(gold_tc[key], abs=1e-4), f"Mismatch in {key}"

    # 8. Course Quality Tables
    cq = results["course_quality"]
    by_cat = cq["by_category"].to_dict(orient="records")
    for row, gold_row in zip(by_cat, golden_baseline["by_category"], strict=True):
        assert row["CourseCategory"] == gold_row["CourseCategory"]
        assert row["avg_rating"] == pytest.approx(gold_row["avg_rating"], abs=1e-3)
        assert row["count"] == gold_row["count"]

    by_lvl = cq["by_level"].to_dict(orient="records")
    for row, gold_row in zip(by_lvl, golden_baseline["by_level"], strict=True):
        assert row["CourseLevel"] == gold_row["CourseLevel"]
        assert row["avg_rating"] == pytest.approx(gold_row["avg_rating"], abs=1e-3)
        assert row["count"] == gold_row["count"]

    cat_pivot = cq["cat_level_pivot"].to_dict(orient="index")
    for cat, levels in golden_baseline["cat_level_pivot"].items():
        assert cat in cat_pivot
        for lvl, val in levels.items():
            comp_val = cat_pivot[cat][lvl]
            if math.isnan(val):
                assert math.isnan(comp_val)
            else:
                assert comp_val == pytest.approx(val, abs=1e-3)
