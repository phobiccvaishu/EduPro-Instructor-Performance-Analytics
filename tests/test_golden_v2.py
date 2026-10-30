"""
tests.test_golden_v2
====================
Golden V2 snapshot regression suite:
Validates byte-for-byte or 1e-4 tolerance consistency of all modern v2 analytical outputs,
entity tables, findings F1-F13, and KPIs v2 against tests/golden_v2.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edupro.analytics.engine import AnalyticsEngine
from edupro.config import AnalysisConfig


@pytest.fixture
def golden_v2() -> dict:
    """Load golden v2 snapshot file."""
    path = Path(__file__).resolve().parent / "golden_v2.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_v2_outputs_match_golden_snapshot(dataset, golden_v2):
    """Confirm all modern v2 analytical outputs, findings, and KPIs match the golden snapshot."""
    engine = AnalyticsEngine(
        dataset.master,
        dataset.teachers,
        dataset.courses,
        dataset.transactions,
    )
    # Run in fast mode with fixed seed for deterministic comparison
    cfg = AnalysisConfig(fast_mode=True, seed=42)
    results = engine.run_full_analysis(cfg=cfg)

    # 1. Root structure verification
    assert "entity_tables" in results
    assert "findings" in results
    assert "kpis_v2" in results
    assert "meta" in results

    # Ensure deprecated legacy keys are absent
    legacy_keys = [
        "kpis",
        "experience_vs_performance",
        "course_quality",
        "instructor_impact",
        "expertise_performance",
        "teacher_course_corr",
    ]
    for k in legacy_keys:
        assert k not in results, f"Deprecated legacy key {k} still present in results"

    # 2. Entity table row counts
    tt = results["entity_tables"]["teachers"]
    ct = results["entity_tables"]["courses"]
    assert len(tt) == 60
    assert len(ct) == 60
    assert tt["total_enrollments"].sum() == 10000
    assert ct["total_enrollments"].sum() == 10000

    # 3. KPIs v2 match snapshot
    gold_kpis = golden_v2["kpis_v2"]
    for kpi_key in [
        "avg_teacher_rating",
        "avg_course_rating",
        "rating_consistency_index",
        "experience_impact",
        "enrollment_influence",
    ]:
        actual_val = results["kpis_v2"][kpi_key].value
        expected_val = gold_kpis[kpi_key]["value"]
        assert actual_val == pytest.approx(expected_val, abs=1e-4), f"Mismatch in KPI {kpi_key}"

    # 4. Findings F1-F13 match snapshot
    gold_findings = golden_v2["findings"]
    for fid in ["F1", "F2", "F3", "F4a", "F4b", "F5", "F6", "F7", "F8a", "F10", "F11", "F13"]:
        actual_finding = results["findings"][fid]
        expected_finding = gold_findings[fid]

        assert actual_finding.verdict == expected_finding["verdict"], f"Verdict mismatch in {fid}"
        assert actual_finding.n == expected_finding["n"], f"Sample size n mismatch in {fid}"
        assert actual_finding.statistic == pytest.approx(expected_finding["statistic"], abs=1e-3), (
            f"Statistic mismatch in {fid}"
        )
