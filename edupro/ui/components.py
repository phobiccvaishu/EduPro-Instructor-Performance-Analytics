"""
edupro.ui.components
====================
Reusable UI component primitives for the EduPro executive dashboard.
Strictly non-decorative, emoji-free, accessible, and driven by analytical metadata.
"""

from __future__ import annotations

import math
from typing import Any

import streamlit as st

from edupro.analytics.kpis import KPIResult
from edupro.formatting import fmt_half_up
from edupro.ui.theme import (
    COLOR_BORDER,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    STATUS_COLORS,
)


def section_header(title: str, subtitle: str | None = None) -> None:
    """Render a clean, unadorned executive section header with optional subtitle."""
    st.markdown(
        f"""
<div style='margin: 1.25rem 0 0.4rem 0;'>
    <div style='font-size: 1.15rem; font-weight: 700; color: {COLOR_TEXT_PRIMARY};'>{title}</div>
    {f"<div style='font-size: 0.85rem; color: {COLOR_TEXT_SECONDARY}; margin-top: 0.15rem;'>{subtitle}</div>" if subtitle else ""}
</div>
""",
        unsafe_allow_html=True,
    )


def scope_caption(
    view: Any,
    unit: str = "instructor & course",
    is_fast_mode: bool = False,
) -> None:
    """Render a standardized metadata caption showing observational scope and active sample sizes."""
    n_t = view.counts.get("n_teachers_active", view.counts.get("n_teachers", 0))
    n_c = view.counts.get("n_courses_active", view.counts.get("n_courses", 0))
    n_p = view.counts.get("n_pairs", 0)
    n_e = view.counts.get("n_enrollments", 0)

    fast_badge = " · Fast estimates active (300 bootstrap iterations)" if is_fast_mode else ""

    caption_text = (
        f"Unit of analysis: {unit} | Scope: {n_t:,} instructors, {n_c:,} courses "
        f"({n_p:,} transaction pairs, {n_e:,} enrollments){fast_badge}"
    )
    st.caption(caption_text)


def evidence_badge(verdict: str, evidence_level: str | None = None) -> str:
    """Format an HTML inline text badge indicating statistical verdict and evidence grade."""
    style = STATUS_COLORS.get(verdict, STATUS_COLORS["Not supported"])
    label = f"{verdict}"
    if evidence_level and evidence_level not in ("None", "Descriptive"):
        label += f" · Evidence: {evidence_level}"

    return (
        f"<span style='display: inline-block; padding: 0.15rem 0.5rem; font-size: 0.75rem; "
        f"font-weight: 600; color: {style['text']}; background-color: {style['bg']}; "
        f"border: 1px solid {style['border']}; border-radius: 3px; letter-spacing: 0.02em;'>"
        f"{label}</span>"
    )


def kpi_tile(
    col: Any,
    label: str,
    kpi: KPIResult,
    delta: str | None = None,
    delta_color: str = "off",
    format_spec: str = ".3f",
    unit_suffix: str = "",
    subcaption: str | None = None,
) -> None:
    """
    Render a single executive KPI tile displaying the point estimate,
    95% confidence interval, sample size, and caveat tooltip.
    """
    val = kpi.value
    decimals = 3
    add_sign = False
    spec_clean = format_spec.lstrip("+")
    if format_spec.startswith("+"):
        add_sign = True
    if spec_clean.endswith("f"):
        try:
            decimals = int(spec_clean.replace(".", "").replace("f", ""))
        except Exception:
            decimals = 3

    if math.isnan(val):
        display_val = "Insufficient data"
    else:
        display_val = fmt_half_up(
            val, decimals=decimals, suffix=unit_suffix, add_sign=add_sign
        )

    # Sub-caption with 95% CI and n
    if subcaption is not None:
        meta_subtext = subcaption
    else:
        ci_str = ""
        if not (math.isnan(kpi.ci_low) or math.isnan(kpi.ci_high)):
            str_low = fmt_half_up(kpi.ci_low, decimals=decimals)
            str_high = fmt_half_up(kpi.ci_high, decimals=decimals)
            ci_str = f" [95% CI: {str_low}, {str_high}]"
        meta_subtext = f"n = {kpi.n}{ci_str}"

    # Build comprehensive help tooltip from definition and caveat
    help_parts = []
    if kpi.definition:
        help_parts.append(kpi.definition)
    if kpi.caveat:
        help_parts.append(f"Caveat: {kpi.caveat}")
    if kpi.evidence and kpi.evidence != "None":
        help_parts.append(f"Evidence Grade: {kpi.evidence}")
    help_tooltip = "\n\n".join(help_parts) if help_parts else None

    # Render metric in column
    col.metric(
        label=label,
        value=display_val,
        delta=delta,
        delta_color=delta_color,
        help=help_tooltip,
    )

    # Sub-caption under metric tile
    col.caption(meta_subtext)


def insight_box(
    headline: str,
    evidence_text: str,
    caveat_text: str | None = None,
    verdict: str | None = None,
    evidence_level: str | None = None,
    stat_meta: str | None = None,
) -> None:
    """Render a structured executive insight callout derived directly from finding objects."""
    badge_html = evidence_badge(verdict, evidence_level) if verdict else ""
    meta_html = (
        f"<span style='font-size: 0.80rem; color: {COLOR_TEXT_MUTED}; margin-left: 0.6rem;'>{stat_meta}</span>"
        if stat_meta
        else ""
    )

    st.markdown(
        f"""
<div style='background-color: #F8FAFC; border: 1px solid {COLOR_BORDER}; border-left: 4px solid #1E40AF; padding: 0.85rem 1rem; border-radius: 3px; margin: 0.6rem 0 1rem 0;'>
    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;'>
        <div style='font-size: 0.95rem; font-weight: 700; color: {COLOR_TEXT_PRIMARY};'>{headline}</div>
        <div>{badge_html}{meta_html}</div>
    </div>
    <div style='font-size: 0.86rem; color: {COLOR_TEXT_SECONDARY}; line-height: 1.45;'>{evidence_text}</div>
    {f"<div style='font-size: 0.80rem; color: {COLOR_TEXT_MUTED}; margin-top: 0.35rem; font-style: italic;'>Note: {caveat_text}</div>" if caveat_text else ""}
</div>
""",
        unsafe_allow_html=True,
    )


def method_note(title: str, items: list[tuple[str, str]]) -> None:
    """Render a collapsible methodology and caveats block for statistical defensibility."""
    with st.expander(f"Methodology & Governance: {title}", expanded=False):
        for label, desc in items:
            st.markdown(
                f"<div style='margin-bottom: 0.35rem; font-size: 0.84rem;'>"
                f"<strong style='color: {COLOR_TEXT_PRIMARY};'>{label}:</strong> "
                f"<span style='color: {COLOR_TEXT_SECONDARY};'>{desc}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


def empty_state(message: str) -> None:
    """Render an informative guidance box when filters return zero active records."""
    st.info(f"Notice: {message}")
