"""
tests.test_stats
================
Unit tests for statistical helper functions in edupro.analytics.stats.
"""

from __future__ import annotations

import numpy as np
import pytest

from edupro.analytics.stats import (
    bootstrap_ci,
    correlation,
    evidence_label,
    gini,
    hhi,
    holm_adjust,
    kruskal_with_effect,
    mann_whitney_with_effect,
    top_n_share,
)


class TestCorrelation:
    """Tests for Pearson and Spearman correlation functions."""

    def test_pearson_known_values(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        res = correlation(x, y, method="pearson")
        assert res["status"] == "ok"
        assert pytest.approx(res["statistic"], rel=1e-4) == 1.0
        assert pytest.approx(res["p_value"], abs=1e-5) == 0.0
        assert res["ci_low"] > 0.99
        assert res["effect"] == "large"

    def test_spearman_monotonic(self) -> None:
        x = [1, 2, 3, 4, 5, 6, 7]
        y = [1, 4, 9, 16, 25, 36, 49]
        res = correlation(x, y, method="spearman", n_boot=200, seed=20260601)
        assert res["status"] == "ok"
        assert pytest.approx(res["statistic"], rel=1e-4) == 1.0
        assert res["ci_low"] > 0.95

    def test_empty_and_constant_inputs(self) -> None:
        res_empty = correlation([], [], method="pearson")
        assert res_empty["status"] == "insufficient data"
        assert np.isnan(res_empty["statistic"])

        res_const = correlation([1, 1, 1], [2, 3, 4], method="pearson")
        assert res_const["status"] == "insufficient data"

        res_short = correlation([1, 2], [2, 3], method="pearson")
        assert res_short["status"] == "insufficient data"


class TestBootstrapCI:
    """Tests for deterministic bootstrap confidence intervals."""

    def test_bootstrap_determinism(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ci1 = bootstrap_ci(data, np.mean, n_boot=500, seed=42)
        ci2 = bootstrap_ci(data, np.mean, n_boot=500, seed=42)
        assert pytest.approx(ci1[0], abs=1e-5) == ci2[0]
        assert pytest.approx(ci1[1], abs=1e-5) == ci2[1]
        assert ci1[0] < np.mean(data) < ci1[1]

    def test_cluster_bootstrap(self) -> None:
        import pandas as pd

        df = pd.DataFrame(
            {
                "val": [10.0, 12.0, 100.0, 102.0],
                "cid": ["A", "A", "B", "B"],
            }
        )
        ci = bootstrap_ci(
            df, lambda d: float(d["val"].mean()), n_boot=200, seed=42, cluster_ids=df["cid"]
        )
        assert not np.isnan(ci[0])
        assert not np.isnan(ci[1])


class TestNonParametricTests:
    """Tests for Kruskal-Wallis and Mann-Whitney U with effect sizes."""

    def test_kruskal_wallis(self) -> None:
        g1 = [1, 2, 3, 4, 5]
        g2 = [10, 11, 12, 13, 14]
        g3 = [20, 21, 22, 23, 24]
        res = kruskal_with_effect(g1, g2, g3)
        assert res["status"] == "ok"
        assert res["p_value"] < 0.01
        assert res["epsilon_squared"] > 0.5
        assert res["effect"] == "large"

    def test_mann_whitney(self) -> None:
        x = [1, 2, 3, 4, 5]
        y = [6, 7, 8, 9, 10]
        res = mann_whitney_with_effect(x, y)
        assert res["status"] == "ok"
        assert res["p_value"] < 0.05
        assert abs(res["rank_biserial"]) > 0.5


class TestMultipleTesting:
    """Tests for Holm-Bonferroni correction."""

    def test_holm_adjust(self) -> None:
        # Known p-values
        raw_p = [0.01, 0.04, 0.03, 0.005]
        # Sorted: 0.005 (x4 = 0.02), 0.01 (x3 = 0.03), 0.03 (x2 = 0.06), 0.04 (x1 = 0.04 -> max 0.06)
        adj_p = holm_adjust(raw_p)
        assert pytest.approx(adj_p[3], abs=1e-4) == 0.02
        assert pytest.approx(adj_p[0], abs=1e-4) == 0.03
        assert pytest.approx(adj_p[2], abs=1e-4) == 0.06
        assert pytest.approx(adj_p[1], abs=1e-4) == 0.06


class TestInequalityMetrics:
    """Tests for Gini, Top-N share, and HHI."""

    def test_gini_bounds(self) -> None:
        equal = [10.0, 10.0, 10.0, 10.0]
        assert pytest.approx(gini(equal), abs=1e-5) == 0.0

        unequal = [0.0, 0.0, 0.0, 100.0]
        assert gini(unequal) > 0.70

    def test_top_n_share(self) -> None:
        vals = [10, 20, 30, 40]  # total 100
        assert pytest.approx(top_n_share(vals, 1), abs=1e-5) == 0.40
        assert pytest.approx(top_n_share(vals, 2), abs=1e-5) == 0.70

    def test_hhi(self) -> None:
        monopoly = [100.0]
        assert pytest.approx(hhi(monopoly), abs=1e-4) == 10000.0

        two_equal = [50.0, 50.0]
        assert pytest.approx(hhi(two_equal), abs=1e-4) == 5000.0


class TestEvidenceLabel:
    """Tests for evidence labeling."""

    def test_evidence_labels(self) -> None:
        assert evidence_label(0.001, 60, (0.2, 0.6), "large") == "Strong"
        assert evidence_label(0.001, 20, (0.2, 0.6), "large") == "Moderate"
        assert evidence_label(0.08, 60, (-0.1, 0.4), "small") == "Weak"
        assert evidence_label(0.50, 60, (-0.2, 0.3), "negligible") == "None"
        assert evidence_label(None, 2, None, "undefined") == "Insufficient data"
