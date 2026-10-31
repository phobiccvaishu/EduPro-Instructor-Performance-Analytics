"""
tests.test_kpis_v2
==================
Tests for revised KPI v2 methods and engine pipeline integration.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from edupro.analytics.engine import AnalyticsEngine
from edupro.analytics.kpis import KPIEngine, KPIResult
from edupro.config import AnalysisConfig
from edupro.data.pipeline import load_all


@pytest.fixture(scope="module")
def dataset():
    return load_all()


class TestKPIsV2:
    """Tests for KPIResult structures and values."""

    def test_kpi_v2_results(self, dataset) -> None:
        ke = KPIEngine(dataset.master)
        cfg = AnalysisConfig(fast_mode=True)
        kpis_v2 = ke.compute_all_kpis_v2(dataset.teachers, dataset.courses, cfg=cfg)

        assert len(kpis_v2) >= 5
        for name, res in kpis_v2.items():
            assert isinstance(res, KPIResult)
            assert res.name == name
            assert np.isfinite(res.value)
            assert res.n > 0
            assert res.evidence in ("Strong", "Moderate", "Weak", "None", "Insufficient data")

        # Check dual slope in experience impact
        exp_kpi = kpis_v2["experience_impact"]
        assert "slope_on_teacher_rating" in exp_kpi.extra
        assert "slope_on_course_rating" in exp_kpi.extra
        assert exp_kpi.extra["slope_on_teacher_rating"] > exp_kpi.extra["slope_on_course_rating"]

        # Check robustness flag in enrollment influence
        enr_kpi = kpis_v2["enrollment_influence"]
        assert enr_kpi.extra["robustness"] in ("Robust", "Fragile")

    def test_pipeline_integration_and_performance(self, dataset) -> None:
        engine = AnalyticsEngine(
            master=dataset.master,
            teachers_df=dataset.teachers,
            courses_df=dataset.courses,
            transactions_df=dataset.transactions,
        )

        # Fast mode timing (< 2.5s)
        t0 = time.perf_counter()
        fast_res = engine.run_full_analysis(AnalysisConfig(fast_mode=True))
        t_fast = time.perf_counter() - t0
        assert t_fast < 2.5

        # Check presence of all v2 keys
        expected_keys = {
            "entity_tables",
            "findings",
            "kpis_v2",
            "per_instructor_consistency",
            "integration_summary",
            "instructor_profiles",
            "meta",
        }
        assert expected_keys.issubset(set(fast_res.keys()))

        # Sensitivity analysis method
        sens_df = engine.run_sensitivity()
        assert len(sens_df) == 6  # 2x2 grid + 2 robust check rows (HC3 and Spearman)
        assert "attribution" in sens_df.columns
        assert "min_pair_enrollment" in sens_df.columns

    def test_leave_k_out_hand_computed(self) -> None:
        """Item 14: Hand-computed unit test for leave_k_out sensitivity."""
        import pandas as pd

        from edupro.analytics.stats import leave_k_out

        # Synthetic cohort of 4 instructors
        df = pd.DataFrame(
            {
                "TeacherID": ["T1", "T2", "T3", "T4"],
                "TeacherRating": [4.5, 4.2, 3.5, 2.5],
                "total_enrollments": [100, 50, 30, 20],
            }
        )

        def calc_ratio(sub: pd.DataFrame) -> dict[str, float]:
            high_mask = sub["TeacherRating"] >= 4.0
            tot_e = float(sub["total_enrollments"].sum())
            hi_e = float(sub.loc[high_mask, "total_enrollments"].sum())
            n_tot = len(sub)
            n_hi = int(high_mask.sum())
            sh = (hi_e / tot_e) if tot_e > 0 else 0.0
            exp = (n_hi / n_tot) if n_tot > 0 else 1.0
            rat = (sh / exp) if exp > 0 else 0.0
            return {"ratio": rat, "high_share": sh, "n": n_tot}

        res = leave_k_out(calc_ratio, df, sort_col="total_enrollments", k_values=(0, 1, 2))

        # Hand calculations:
        # k=0: total_e=200, hi_e=150 (T1+T2), sh=0.75, exp=2/4=0.5 -> ratio = 1.50
        assert res[0]["n"] == 4
        assert res[0]["high_share"] == pytest.approx(0.75)
        assert res[0]["ratio"] == pytest.approx(1.50)

        # k=1: T1 (100 enr) removed. sub has T2(50), T3(30), T4(20).
        # total_e=100, hi_e=50 (T2), sh=0.50, exp=1/3 -> ratio = 0.50 / (1/3) = 1.50
        assert res[1]["n"] == 3
        assert res[1]["high_share"] == pytest.approx(0.50)
        assert res[1]["ratio"] == pytest.approx(1.50)

        # k=2: T1 and T2 removed. sub has T3(30), T4(20).
        # total_e=50, hi_e=0, sh=0.0, exp=0/2=0 -> ratio = 0.0
        assert res[2]["n"] == 2
        assert res[2]["high_share"] == pytest.approx(0.0)
        assert res[2]["ratio"] == pytest.approx(0.0)

    def test_kpi5_p_values(self, dataset) -> None:
        """Item 6: KPI 5 explicitly reports Mann-Whitney U p-value and bootstrap p-value."""
        ke = KPIEngine(dataset.master)
        kpi5 = ke.enrollment_influence_v2(cfg=AnalysisConfig(fast_mode=True, seed=20260601))
        assert "mann_whitney_p_value" in kpi5.extra
        assert "bootstrap_p_value" in kpi5.extra
        assert kpi5.extra["mann_whitney_p_value"] == pytest.approx(0.0301, abs=1e-3)
        assert kpi5.extra["bootstrap_p_value"] <= 0.02

    def test_kpi_cis_deterministic(self, dataset) -> None:
        """Item 22: KPI CIs are fully deterministic when cfg.seed is set."""
        ke = KPIEngine(dataset.master)
        cfg1 = AnalysisConfig(fast_mode=True, seed=42)
        cfg2 = AnalysisConfig(fast_mode=True, seed=42)

        kpis1 = ke.compute_all_kpis_v2(dataset.teachers, dataset.courses, cfg=cfg1)
        kpis2 = ke.compute_all_kpis_v2(dataset.teachers, dataset.courses, cfg=cfg2)

        for name in kpis1:
            res1 = kpis1[name]
            res2 = kpis2[name]
            assert res1.value == pytest.approx(res2.value, abs=1e-6)
            if np.isfinite(res1.ci_low):
                assert res1.ci_low == pytest.approx(res2.ci_low, abs=1e-6)
                assert res1.ci_high == pytest.approx(res2.ci_high, abs=1e-6)

    def test_default_kpi_cis_match_json(self, dataset) -> None:
        """Item 22: KPI CIs from default config match results/analysis_summary.json within tolerance."""
        import json
        from pathlib import Path

        from edupro.config import default_analysis_config

        summary_path = Path(__file__).resolve().parent.parent / "results" / "analysis_summary.json"
        with open(summary_path) as f:
            summary = json.load(f)
        expected = summary["kpis_v2"]

        ke = KPIEngine(dataset.master)
        kpis = ke.compute_all_kpis_v2(
            dataset.teachers, dataset.courses, cfg=default_analysis_config
        )

        for name, exp_data in expected.items():
            if name in kpis and np.isfinite(exp_data["ci_low"]):
                kpi_res = kpis[name]
                assert kpi_res.ci_low == pytest.approx(exp_data["ci_low"], abs=0.01)
                assert kpi_res.ci_high == pytest.approx(exp_data["ci_high"], abs=0.01)
