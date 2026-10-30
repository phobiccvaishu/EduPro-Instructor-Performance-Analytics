"""
tests.test_inference
====================
Integration tests for hypothesis testing findings F1 through F13.
"""

from __future__ import annotations

import pytest

from edupro.analytics.aggregation import course_table, teacher_table
from edupro.analytics.inference import Finding, run_all_inferences
from edupro.config import AnalysisConfig
from edupro.data.pipeline import load_all


@pytest.fixture(scope="module")
def analysis_findings() -> dict[str, Finding]:
    """Run all inferences once for the module using fast mode."""
    ds = load_all()
    cfg = AnalysisConfig(fast_mode=True)
    tt = teacher_table(ds.teachers, ds.master)
    ct = course_table(ds.courses, ds.master)
    return run_all_inferences(ds.teachers, ds.courses, ds.master, tt, ct, cfg)


class TestInferenceFramework:
    """Tests for all 13 findings."""

    def test_all_13_findings_present(self, analysis_findings: dict[str, Finding]) -> None:
        expected_ids = {f"F{i}" for i in range(1, 14)}
        assert set(analysis_findings.keys()) == expected_ids
        for fid, f in analysis_findings.items():
            assert isinstance(f, Finding)
            assert f.id == fid
            assert f.verdict in ("supported", "not supported", "inconclusive", "insufficient data")
            assert f.evidence in ("Strong", "Moderate", "Weak", "None", "Insufficient data")

    def test_f1_distribution(self, analysis_findings: dict[str, Finding]) -> None:
        f1 = analysis_findings["F1"]
        assert f1.unit_of_analysis == "teacher"
        assert f1.n == 60
        assert 1.0 <= f1.statistic <= 5.0
        assert "shapiro_p" in f1.extra

    def test_f2_experience_teacher_rating(self, analysis_findings: dict[str, Finding]) -> None:
        f2 = analysis_findings["F2"]
        assert f2.unit_of_analysis == "teacher"
        assert f2.n == 60
        assert f2.statistic > 0  # positive correlation
        assert f2.p_value < 0.05
        assert f2.verdict == "supported"
        assert "ols_slope" in f2.extra

    def test_f3_experience_vs_course_rating_contrast(self, analysis_findings: dict[str, Finding]) -> None:
        f3 = analysis_findings["F3"]
        assert f3.unit_of_analysis == "teacher"
        assert "pair_level_cluster_robust" in f3.extra
        pair_cluster = f3.extra["pair_level_cluster_robust"]
        assert "p_value" in pair_cluster
        assert "slope" in pair_cluster

    def test_f4_breakpoint_search(self, analysis_findings: dict[str, Finding]) -> None:
        f4 = analysis_findings["F4"]
        assert "breakpoint_search" in f4.extra
        best_b = f4.extra["breakpoint_search"]["best_breakpoint"]
        assert 3 <= best_b <= 12

    def test_f5_rating_decoupling(self, analysis_findings: dict[str, Finding]) -> None:
        f5 = analysis_findings["F5"]
        assert "rating_decoupling" in f5.extra
        # Decoupling is confirmed by non-significant correlation between TeacherRating and CourseRating
        assert f5.extra["rating_decoupling"] is True

    def test_f6_f7_course_category_and_level(self, analysis_findings: dict[str, Finding]) -> None:
        f6 = analysis_findings["F6"]
        f7 = analysis_findings["F7"]
        assert f6.unit_of_analysis == "course"
        assert f7.unit_of_analysis == "course"
        assert f6.n == 60
        assert f7.n == 60

    def test_f8_gender_analysis(self, analysis_findings: dict[str, Finding]) -> None:
        f8 = analysis_findings["F8"]
        assert "teacher_rating_by_gender" in f8.extra
        assert "cell_level_cluster_bootstraps" in f8.extra

    def test_f9_tier_vs_quality(self, analysis_findings: dict[str, Finding]) -> None:
        f9 = analysis_findings["F9"]
        assert "tier_means" in f9.extra

    def test_f10_rating_vs_enrollment(self, analysis_findings: dict[str, Finding]) -> None:
        f10 = analysis_findings["F10"]
        assert "tier_enrollment_means" in f10.extra

    def test_f11_concentration_and_domain_ols(self, analysis_findings: dict[str, Finding]) -> None:
        f11 = analysis_findings["F11"]
        assert "teacher_gini" in f11.extra
        assert "domain_controlled_ols" in f11.extra
        ols_info = f11.extra["domain_controlled_ols"]
        assert "teacher_rating_coef" in ols_info
        assert "partial_r2" in ols_info

    def test_f12_f13_expertise_and_ml_gap(self, analysis_findings: dict[str, Finding]) -> None:
        f12 = analysis_findings["F12"]
        f13 = analysis_findings["F13"]
        assert "insufficient_domains" in f12.extra
        assert "ml_curriculum_gap" in f13.extra
        ml_gap = f13.extra["ml_curriculum_gap"]
        assert ml_gap["ml_teacher_rating_mean"] > ml_gap["ml_course_rating_catalog_mean"]
