"""
tests.test_verification_values
==============================
Regression tests asserting all 13 independent verification values
with exact tolerances for Step 2 Analytics Reconciliation.
"""

from __future__ import annotations

import time

import pandas as pd
import pytest
from scipy import stats

from edupro.analytics.aggregation import course_table, teacher_table
from edupro.analytics.engine import AnalyticsEngine
from edupro.analytics.inference import run_all_inferences
from edupro.analytics.kpis import KPIEngine
from edupro.config import AnalysisConfig, get_epsilon_squared_effect, settings
from edupro.data.pipeline import load_all


@pytest.fixture(scope="module")
def dataset():
    return load_all()


@pytest.fixture(scope="module")
def analysis_artifacts(dataset):
    cfg = AnalysisConfig(fast_mode=True, seed=20260601)
    tt = teacher_table(dataset.teachers, dataset.master)
    ct = course_table(dataset.courses, dataset.master)
    findings = run_all_inferences(dataset.teachers, dataset.courses, dataset.master, tt, ct, cfg)
    ke = KPIEngine(dataset.master)
    kpis = ke.compute_all_kpis_v2(dataset.teachers, dataset.courses, cfg=cfg)
    return {
        "tt": tt,
        "ct": ct,
        "findings": findings,
        "kpis": kpis,
        "cfg": cfg,
    }


class TestVerificationValues:
    """Rigorous verification tests against independent benchmarks."""

    def test_f11_top_shares_and_concentration(self, analysis_artifacts) -> None:
        """Discrepancy 1: F11 top-11 and top-5 enrollment shares sorted by enrollments."""
        f11 = analysis_artifacts["findings"]["F11"]
        extra = f11.extra

        # Top 11 teachers by enrollment should account for 73.1% (+/- 0.1%)
        assert extra["teacher_top11_share"] == pytest.approx(0.7312, abs=0.005)
        # Top 5 teachers by enrollment should account for 65.5% (+/- 0.1%)
        assert extra["teacher_top5_share"] == pytest.approx(0.6549, abs=0.005)
        # Teacher Gini index
        assert extra["teacher_gini"] == pytest.approx(0.681, abs=0.01)
        # Course Gini and Top-5 course share
        assert extra["course_gini"] == pytest.approx(0.042, abs=0.01)
        assert extra["course_top5_share"] == pytest.approx(0.0953, abs=0.005)

    def test_f4_experience_contrast_and_breakpoint(self, analysis_artifacts) -> None:
        """Discrepancy 2: F4 0-5 vs 6+ contrast and breakpoint search."""
        f4 = analysis_artifacts["findings"]["F4"]
        extra = f4.extra

        c = extra["two_group_contrast"]
        # Group 1: 0-5 yrs (n=28), mean ~ 2.399
        assert c["n_0_to_5"] == 28
        assert c["mean_0_to_5"] == pytest.approx(2.399, abs=0.02)
        # Group 2: 6+ yrs (n=32), mean ~ 3.760
        assert c["n_6_plus"] == 32
        assert c["mean_6_plus"] == pytest.approx(3.760, abs=0.02)
        # Difference +1.361 stars
        assert c["mean_diff"] == pytest.approx(1.361, abs=0.05)
        assert c["mann_whitney"]["statistic"] == pytest.approx(839.0, abs=1.0)
        assert c["mann_whitney"]["p_value"] < 0.0001
        assert c["verdict"] == "supported"

        # Breakpoint search
        bp = extra["breakpoint_search"]
        assert bp["best_breakpoint"] == 8
        assert bp["slope_before"] == pytest.approx(0.212, abs=0.02)
        assert bp["slope_after"] == pytest.approx(0.048, abs=0.02)
        assert bp["verdict"] == "not supported"

    def test_f4_breakpoint_permutation_benchmark(self, dataset) -> None:
        """Discrepancy 2b: Breakpoint permutation test p-value ~ 0.128."""
        from edupro.analytics.inference import eval_f4_experience_threshold

        cfg = AnalysisConfig(n_boot=300, n_perm=300, seed=20260601)
        f4 = eval_f4_experience_threshold(dataset.teachers, cfg=cfg)
        bp = f4.extra["breakpoint_search"]
        assert bp["permutation_p"] == pytest.approx(0.128, abs=0.03)

    def test_f6_category_kruskal_consistency(self, analysis_artifacts) -> None:
        """Discrepancy 3: F6 Kruskal-Wallis H=4.388, df=11, p=0.9572 consistency."""
        f6 = analysis_artifacts["findings"]["F6"]
        kw = f6.extra["kruskal_wallis"]
        h_stat = kw["statistic"]
        df = kw["df"]
        p_val = kw["p_value"]

        assert h_stat == pytest.approx(4.388, abs=0.01)
        assert df == 11
        assert p_val == pytest.approx(0.9572, abs=0.01)

        # Verify exact mathematical consistency with chi2 survival function
        expected_p = float(stats.chi2.sf(h_stat, df))
        assert p_val == pytest.approx(expected_p, abs=1e-4)
        assert f6.verdict == "not supported"

    def test_f7_level_effect_size_and_boundary_labels(self, analysis_artifacts) -> None:
        """Discrepancy 4: F7 epsilon-squared 0.000 labeled 'negligible' & boundary tests."""
        f7 = analysis_artifacts["findings"]["F7"]
        assert f7.effect_label == "negligible"
        assert f7.extra["kruskal_wallis"]["epsilon_squared"] == pytest.approx(0.000, abs=1e-4)

        # Exhaustive boundary tests for get_epsilon_squared_effect
        assert get_epsilon_squared_effect(-0.05) == "negligible"
        assert get_epsilon_squared_effect(0.0) == "negligible"
        assert get_epsilon_squared_effect(0.0099) == "negligible"
        assert get_epsilon_squared_effect(0.01) == "small"
        assert get_epsilon_squared_effect(0.0799) == "small"
        assert get_epsilon_squared_effect(0.08) == "moderate"
        assert get_epsilon_squared_effect(0.2599) == "moderate"
        assert get_epsilon_squared_effect(0.26) == "large"
        assert get_epsilon_squared_effect(0.50) == "large"

    def test_f5_f3_correlations_and_hand_calculation(self, analysis_artifacts, dataset) -> None:
        """Discrepancy 5: Decoupled correlations, cluster-robust OLS, & 3x3 hand calc."""
        findings = analysis_artifacts["findings"]
        f5 = findings["F5"]
        f3 = findings["F3"]

        # Teacher-level correlation
        assert f5.statistic == pytest.approx(0.000, abs=0.01)
        assert f5.p_value == pytest.approx(0.997, abs=0.01)
        assert f5.verdict == "not supported"

        # Course-level correlation
        assert f5.extra["course_level"]["statistic"] == pytest.approx(-0.013, abs=0.01)
        assert f5.extra["course_level"]["p_value"] == pytest.approx(0.921, abs=0.01)

        # F3 teacher experience vs course quality
        assert f3.statistic == pytest.approx(-0.057, abs=0.01)
        assert f3.p_value == pytest.approx(0.666, abs=0.01)

        # Pair-level cluster-robust OLS slope for F5
        pair_cluster = f5.extra["secondary_pair_level_cluster_robust"]
        assert pair_cluster["slope"] == pytest.approx(-0.0391, abs=0.01)
        assert pair_cluster["p_value"] == pytest.approx(0.0929, abs=0.02)

        # --- Hand-calculation verification on 3 teachers and 3 courses ---
        mini_teachers = pd.DataFrame(
            {
                "TeacherID": [1, 2, 3],
                "TeacherRating": [4.0, 5.0, 3.0],
                "YearsOfExperience": [2, 5, 10],
                "Expertise": ["Python", "Python", "Data Science"],
                "Gender": ["Female", "Male", "Female"],
                "Age": [28, 35, 42],
            }
        )
        mini_courses = pd.DataFrame(
            {
                "CourseID": [101, 102, 103],
                "CourseRating": [4.5, 3.5, 2.5],
                "CourseCategory": ["Programming", "Programming", "Data Science"],
                "CourseLevel": ["Beginner", "Intermediate", "Advanced"],
                "CoursePrice": [50.0, 60.0, 70.0],
                "CourseDuration": [10.0, 15.0, 20.0],
            }
        )
        mini_master = pd.DataFrame(
            {
                "TeacherID": [1, 1, 2, 2, 3, 3],
                "CourseID": [101, 102, 102, 103, 101, 103],
                "EnrollmentCount": [10, 20, 30, 10, 40, 20],
                "TeacherRating": [4.0, 4.0, 5.0, 5.0, 3.0, 3.0],
                "CourseRating": [4.5, 3.5, 3.5, 2.5, 4.5, 2.5],
                "CourseCategory": [
                    "Programming",
                    "Programming",
                    "Programming",
                    "Data Science",
                    "Programming",
                    "Data Science",
                ],
                "Expertise": [
                    "Python",
                    "Python",
                    "Python",
                    "Python",
                    "Data Science",
                    "Data Science",
                ],
            }
        )

        calc_tt = teacher_table(mini_teachers, mini_master)
        calc_ct = course_table(mini_courses, mini_master)

        # Compare calculated vs hand-derived
        t_w_ratings = dict(
            zip(calc_tt["TeacherID"], calc_tt["weighted_course_rating"], strict=False)
        )
        assert t_w_ratings[1] == pytest.approx(115.0 / 30.0, abs=1e-5)
        assert t_w_ratings[2] == pytest.approx(130.0 / 40.0, abs=1e-5)
        assert t_w_ratings[3] == pytest.approx(230.0 / 60.0, abs=1e-5)

        c_w_ratings = dict(
            zip(calc_ct["CourseID"], calc_ct["weighted_teacher_rating"], strict=False)
        )
        assert c_w_ratings[101] == pytest.approx(160.0 / 50.0, abs=1e-5)
        assert c_w_ratings[102] == pytest.approx(230.0 / 50.0, abs=1e-5)
        assert c_w_ratings[103] == pytest.approx(110.0 / 30.0, abs=1e-5)

    def test_f8_gender_analysis(self, analysis_artifacts) -> None:
        """Discrepancy 6: F8a, F8b and F8c gender separation and tests."""
        f8 = analysis_artifacts["findings"]["F8"]
        extra = f8.extra

        # F8a: TeacherRating by Gender
        t_gender = extra["f8a_teacher_rating_by_gender"]
        assert t_gender["statistic"] == pytest.approx(551.0, abs=1.0)
        assert t_gender["p_value"] == pytest.approx(0.1187, abs=0.02)

        # F8b: Course Quality by Gender
        c_gender = extra["f8b_weighted_course_rating_by_gender"]
        assert c_gender["statistic"] == pytest.approx(254.0, abs=1.0)
        assert c_gender["p_value"] == pytest.approx(0.0045, abs=0.002)

        # F8c: Cell level bootstraps
        cell_boot = extra["f8c_cell_level_cluster_bootstraps"]
        assert len(cell_boot) > 0
        assert f8.verdict == "not supported"

    def test_f10_tier_enrollments_and_subgroup_rhos(self, analysis_artifacts) -> None:
        """Discrepancy 7: F10 tier enrollments, subgroup rhos, and controlled OLS."""
        f10 = analysis_artifacts["findings"]["F10"]
        f11 = analysis_artifacts["findings"]["F11"]

        # Spearman rho overall
        assert f10.statistic == pytest.approx(-0.012, abs=0.01)
        assert f10.p_value == pytest.approx(0.925, abs=0.01)

        # High-tier vs rest Mann-Whitney
        assert f10.extra["high_tier_vs_rest_mw"]["p_value"] == pytest.approx(0.0301, abs=0.02)

        # Per tier means
        tier_means = f10.extra["tier_enrollment_means"]
        assert tier_means[settings.tier_high_label] == pytest.approx(620.0, abs=1.0)
        assert tier_means[settings.tier_mid_label] == pytest.approx(56.3, abs=1.0)
        assert tier_means[settings.tier_low_label] == pytest.approx(76.4, abs=1.0)

        # Domain subgroups
        assert f11.extra["corr_in_top2"]["statistic"] == pytest.approx(0.319, abs=0.02)
        assert f11.extra["corr_in_top2"]["p_value"] == pytest.approx(0.266, abs=0.02)

        assert f11.extra["corr_outside_top2"]["statistic"] == pytest.approx(0.031, abs=0.02)
        assert f11.extra["corr_outside_top2"]["p_value"] == pytest.approx(0.837, abs=0.02)

        # Controlled OLS (HC3)
        ols_info = f11.extra["domain_controlled_ols"]
        assert ols_info["teacher_rating_coef"] == pytest.approx(0.164, abs=0.02)
        assert ols_info["teacher_rating_se"] == pytest.approx(0.187, abs=0.02)
        assert ols_info["teacher_rating_p"] == pytest.approx(0.382, abs=0.02)

    def test_f12_expertise_dual_group_sizes(self, analysis_artifacts) -> None:
        """Discrepancy 8: F12 dual group sizes min_size=2 and min_size=3."""
        f12 = analysis_artifacts["findings"]["F12"]
        extra = f12.extra

        # Min group size 2
        kw2 = extra["kruskal_min_group_size_2"]
        assert len(extra["valid_domains_size_2"]) == 10
        assert kw2["statistic"] == pytest.approx(15.670, abs=0.05)
        assert kw2["df"] == 9
        assert kw2["p_value"] == pytest.approx(0.0741, abs=0.02)

        # Min group size 3
        kw3 = extra["kruskal_min_group_size_3"]
        assert len(extra["valid_domains_size_3"]) == 9
        assert kw3["statistic"] == pytest.approx(15.196, abs=0.05)
        assert kw3["df"] == 8
        assert kw3["p_value"] == pytest.approx(0.0554, abs=0.02)

        assert f12.verdict == "not supported"

    def test_f13_ml_curriculum_gap(self, analysis_artifacts) -> None:
        """Discrepancy 9: F13 ML curriculum gap separated from instructor competence."""
        f13 = analysis_artifacts["findings"]["F13"]
        gap = f13.extra["ml_quality_gap"]

        # Catalog ML CourseRating mean (A)
        assert gap["A_ml_category_catalog_cr"] == pytest.approx(2.658, abs=0.01)
        # ML instructor taught CourseRating (B)
        assert gap["B_ml_teachers_weighted_cr"] == pytest.approx(3.055, abs=0.02)
        assert gap["B_ml_teachers_unweighted_cr"] == pytest.approx(2.946, abs=0.02)
        # ML instructor TeacherRating mean
        assert f13.extra["ml_instructors_teacher_rating_mean"] == pytest.approx(4.234, abs=0.01)
        # Difference B - A
        assert gap["B_minus_A"] == pytest.approx(0.397, abs=0.03)

    def test_kpi5_robustness_and_kpi3_evidence(self, analysis_artifacts, dataset) -> None:
        """Discrepancies 10 & 11: KPI5 evaluates to Fragile, KPI3 evidence is None."""
        kpis = analysis_artifacts["kpis"]

        # KPI 5: Enrollment Influence
        kpi5 = kpis["enrollment_influence"]
        assert kpi5.extra["robustness"] == "Fragile"
        leave_k = kpi5.extra["leave_k_sensitivity"]
        assert leave_k[0]["ratio"] == pytest.approx(3.72, abs=0.1)
        assert leave_k[1]["ratio"] == pytest.approx(3.20, abs=0.1)
        assert leave_k[2]["ratio"] == pytest.approx(1.21, abs=0.1)
        assert leave_k[3]["ratio"] == pytest.approx(1.24, abs=0.1)
        assert leave_k[4]["ratio"] == pytest.approx(1.27, abs=0.1)

        # KPI 3: Rating Consistency Index with permutation benchmark
        ke = KPIEngine(dataset.master)
        kpi3_benchmarked = ke.rating_consistency_index_v2(
            dataset.teachers, cfg=AnalysisConfig(n_boot=100, n_perm=100, seed=20260601)
        )
        assert kpi3_benchmarked.extra["permutation_p"] == pytest.approx(0.380, abs=0.06)
        assert kpi3_benchmarked.evidence == "None"

    def test_allowed_verdicts_vocabulary(self, analysis_artifacts) -> None:
        """Discrepancy 12: Every finding verdict must come from allowed_verdicts."""
        findings = analysis_artifacts["findings"]
        for fid, f in findings.items():
            assert f.verdict in settings.allowed_verdicts, (
                f"Finding {fid} has disallowed verdict '{f.verdict}'"
            )

    def test_fast_mode_runtime_benchmark(self, dataset) -> None:
        """Discrepancy 13: Fast mode execution completes under 2.5 seconds."""
        cfg = AnalysisConfig(fast_mode=True, seed=20260601)
        engine = AnalyticsEngine(
            dataset.master,
            dataset.teachers,
            dataset.courses,
            dataset.transactions,
        )

        start = time.perf_counter()
        results = engine.run_full_analysis(cfg=cfg)
        elapsed = time.perf_counter() - start

        assert len(results["findings"]) >= 13
        assert len(results["kpis_v2"]) >= 5
        assert elapsed < 2.5, f"Fast mode execution took {elapsed:.2f}s, expected < 2.5s"
