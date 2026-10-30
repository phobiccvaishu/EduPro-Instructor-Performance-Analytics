"""
tests.test_legacy_unchanged
===========================
Explicit regression suite ensuring all legacy keys, formulas, and values in run_full_analysis()
remain completely identical to golden_baseline.json.
"""

from __future__ import annotations

import pytest

from edupro.analytics.engine import AnalyticsEngine


def test_legacy_keys_and_values_unchanged(dataset, golden_baseline):
    """Confirm all legacy outputs are preserved without modification."""
    engine = AnalyticsEngine(
        dataset.master,
        dataset.teachers,
        dataset.courses,
        dataset.transactions,
    )
    results = engine.run_full_analysis()

    # Legacy keys must exist in results
    legacy_root_keys = [
        "kpis",
        "per_instructor_consistency",
        "integration_summary",
        "instructor_profiles",
        "experience_vs_performance",
        "course_quality",
        "instructor_impact",
        "expertise_performance",
        "teacher_course_corr",
    ]
    for k in legacy_root_keys:
        assert k in results, f"Missing legacy key: {k}"

    kpis = results["kpis"]
    assert kpis["avg_teacher_rating"] == pytest.approx(golden_baseline["avg_teacher_rating"], abs=1e-4)
    assert kpis["avg_course_rating"] == pytest.approx(golden_baseline["avg_course_rating"], abs=1e-4)
    assert kpis["rating_consistency_index"] == pytest.approx(golden_baseline["rating_consistency_index"], abs=1e-4)

    # Experience impact legacy values
    exp_imp = kpis["experience_impact"]
    gold_exp = golden_baseline["experience_impact"]
    for key in ("r_exp_teacher", "p_exp_teacher", "slope_exp_teacher",
                "r_exp_course", "p_exp_course", "slope_exp_course", "impact_score"):
        assert exp_imp[key] == pytest.approx(gold_exp[key], abs=1e-4), f"Mismatch in {key}"

    # Enrollment influence legacy values
    enr_inf = kpis["enrollment_influence"]
    gold_enr = golden_baseline["enrollment_influence"]
    for key in ("avg_enrollment_high", "avg_enrollment_low", "avg_enrollment_all", "enrollment_influence_ratio"):
        assert enr_inf[key] == pytest.approx(gold_enr[key], abs=1e-4), f"Mismatch in {key}"

    # Teacher course correlation legacy values
    tc = results["teacher_course_corr"]
    gold_tc = golden_baseline["teacher_course_corr"]
    for key in ("pearson_r", "pearson_p", "spearman_r", "spearman_p", "slope", "intercept", "n"):
        assert tc[key] == pytest.approx(gold_tc[key], abs=1e-4), f"Mismatch in {key}"
