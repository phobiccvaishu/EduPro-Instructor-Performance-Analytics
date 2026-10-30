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

        assert len(kpis_v2) == 5
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

        # Check presence of all legacy and new keys
        expected_keys = {
            "kpis",
            "per_instructor_consistency",
            "integration_summary",
            "instructor_profiles",
            "experience_vs_performance",
            "course_quality",
            "instructor_impact",
            "expertise_performance",
            "teacher_course_corr",
            "entity_tables",
            "findings",
            "kpis_v2",
            "meta",
        }
        assert expected_keys.issubset(set(fast_res.keys()))

        # Sensitivity analysis method
        sens_df = engine.run_sensitivity()
        assert len(sens_df) == 4  # 2x2 grid
        assert "attribution" in sens_df.columns
        assert "min_pair_enrollment" in sens_df.columns
