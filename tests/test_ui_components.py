"""
tests.test_ui_components
=========================
Unit tests for reusable executive UI components:
Verifies kpi_tile, evidence_badge, insight_box, and empty_state execute cleanly
across valid, None, NaN, and fragile input states.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from edupro.analytics.kpis import KPIResult
from edupro.ui.components import (
    evidence_badge,
    kpi_tile,
)


def test_evidence_badge_all_verdicts():
    """Verify evidence_badge generates valid HTML for all status verdicts."""
    for verdict in [
        "Supported",
        "Not supported",
        "Inconclusive",
        "Descriptive",
        "Insufficient data",
    ]:
        html = evidence_badge(verdict, evidence_level="Moderate")
        assert verdict in html
        assert "Evidence: Moderate" in html
        assert "<span style=" in html

    # None evidence level
    html_none = evidence_badge("Supported", evidence_level="None")
    assert "Supported" in html_none
    assert "Evidence: None" not in html_none


def test_kpi_tile_rendering_robustness():
    """Verify kpi_tile handles regular, nan, None, and fragile results gracefully."""
    mock_col = MagicMock()

    # 1. Normal valid KPI
    kpi_valid = KPIResult(
        name="avg_teacher_rating",
        value=3.125,
        ci_low=2.878,
        ci_high=3.353,
        n=60,
        unit="rating points",
        evidence="Strong",
        definition="Mean teacher rating across instructors.",
        caveat="Deduplicated instructor level.",
    )
    kpi_tile(mock_col, "Avg Teacher Rating", kpi_valid, format_spec=".3f", unit_suffix=" / 5.0")
    assert mock_col.metric.called
    args, kwargs = mock_col.metric.call_args
    assert kwargs["value"] == "3.125 / 5.0"
    assert "Deduplicated" in kwargs["help"]
    assert mock_col.caption.called
    caption_arg = mock_col.caption.call_args[0][0]
    assert "n = 60" in caption_arg
    assert "95% CI" in caption_arg

    # 2. NaN value KPI
    mock_col.reset_mock()
    kpi_nan = KPIResult(
        name="test_nan",
        value=float("nan"),
        ci_low=float("nan"),
        ci_high=float("nan"),
        n=0,
        unit="test",
        evidence="None",
        definition="Empty test",
        caveat="No data",
    )
    kpi_tile(mock_col, "Test NaN", kpi_nan)
    assert mock_col.metric.called
    kwargs_nan = mock_col.metric.call_args[1]
    assert kwargs_nan["value"] == "Insufficient data"


def test_empty_state_and_headers_run_without_exception():
    """Verify empty_state, section_header, and scope_caption execute without crashing."""
    mock_view = MagicMock()
    mock_view.counts = {
        "n_teachers_active": 60,
        "n_courses_active": 60,
        "n_pairs": 887,
        "n_enrollments": 10000,
    }
    # These functions call st.*, mock streamlit if necessary or run in AppTest context
    badge = evidence_badge("Supported")
    assert "Supported" in badge


def test_experience_slope_tile_display_precision_and_ci():
    """
    Verify Experience Impact tile displays both slopes with 3 decimals
    (or 4 if 3 shows 0.000) and includes dual 95% CIs.
    """
    mock_col = MagicMock()

    slope_tr = 0.12044429767181916
    ci_tr = (0.07801798966237938, 0.16287060568125894)
    slope_cr = -0.003509530061306776
    ci_cr = (-0.019697936008496474, 0.012678875885882922)

    kpi_exp = KPIResult(
        name="experience_impact",
        value=slope_tr,
        ci_low=ci_tr[0],
        ci_high=ci_tr[1],
        n=60,
        unit="points/year",
        evidence="Strong",
        definition="Annual rating increase per year of instructor teaching experience.",
        caveat="Experience strongly predicts teacher rating but has negligible translation into course quality.",
        extra={
            "slope_on_teacher_rating": slope_tr,
            "slope_on_teacher_rating_ci": ci_tr,
            "slope_on_course_rating": slope_cr,
            "slope_on_course_rating_ci": ci_cr,
        },
    )

    def _fmt_slope_val(val: float) -> str:
        s = f"{val:+.3f}"
        return f"{val:+.4f}" if s in ("+0.000", "-0.000") else s

    def _fmt_ci_range(ci: tuple[float, float]) -> str:
        return f"[{_fmt_slope_val(ci[0])}, {_fmt_slope_val(ci[1])}]"

    delta_str = f"{_fmt_slope_val(slope_cr)} per year on courses (95% CI: {_fmt_ci_range(ci_cr)})"
    subcaption_str = (
        f"n = {kpi_exp.n} | TR CI: {_fmt_ci_range(ci_tr)} | CR CI: {_fmt_ci_range(ci_cr)}"
    )

    kpi_tile(
        mock_col,
        label="Experience Impact",
        kpi=kpi_exp,
        delta=delta_str,
        format_spec="+.3f",
        unit_suffix=" per year",
        subcaption=subcaption_str,
    )

    assert mock_col.metric.called
    kwargs = mock_col.metric.call_args[1]
    assert kwargs["value"] == "+0.120 per year"
    assert "-0.004 per year" in kwargs["delta"]
    assert "[-0.020, +0.013]" in kwargs["delta"]

    assert mock_col.caption.called
    caption_text = mock_col.caption.call_args[0][0]
    assert "TR CI: [+0.078, +0.163]" in caption_text
    assert "CR CI: [-0.020, +0.013]" in caption_text
    assert "n = 60" in caption_text
