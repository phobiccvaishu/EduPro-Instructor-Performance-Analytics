"""
app.py — EduPro Instructor & Course Quality Analytics Platform
=============================================================
Executive decision-support dashboard for instructor performance evaluation
and course content quality analysis.

Run:
    python -m streamlit run app.py
"""

from __future__ import annotations

import math
from typing import Any

import streamlit as st

from edupro.data.pipeline import load_all
from edupro.filters import FilterState, apply_filters
from edupro.logging_setup import get_logger
from edupro.ui.components import empty_state, kpi_tile
from edupro.ui.tabs import (
    tab1_rankings,
    tab2_experience,
    tab3_course_quality,
    tab4_domain,
    tab5_relationships,
)
from edupro.ui.theme import inject_theme_css
from edupro.ui_filters import render_sidebar_filters

logger = get_logger("app")

# ── 1. Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduPro Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject executive styling
inject_theme_css()


# ── 2. Data & Analytics Loading (Cached) ───────────────────────────────────────
@st.cache_data(show_spinner="Loading EduPro dataset...")
def load_data():
    """Load and validate the EduPro multi-sheet dataset."""
    return load_all()


@st.cache_data(show_spinner="Computing analytics on filtered view...")
def run_analytics_cached(data_hash: str, filter_dict: dict[str, Any]):
    """Execute analytics pipeline on active filtered subset."""
    from edupro.analytics.engine import AnalyticsEngine
    from edupro.config import AnalysisConfig

    dataset = load_all()
    state = FilterState.from_dict(filter_dict)
    view = apply_filters(dataset, state)
    if view.is_empty:
        return {"empty": True}

    cfg = AnalysisConfig(fast_mode=True)
    engine = AnalyticsEngine(
        view.master,
        view.teachers,
        view.courses,
        view.transactions,
    )
    res = engine.run_full_analysis(cfg=cfg)
    res["empty"] = False
    return res


# ── 3. Application State & Filtering ──────────────────────────────────────────
try:
    dataset = load_data()
except Exception as e:
    st.error(f"Data loading failed: {e}")
    st.stop()

filter_state = render_sidebar_filters(dataset)
view = apply_filters(dataset, filter_state)

if view.is_empty:
    empty_state(
        "No instructors, courses, or enrollments match the selected filter combination. "
        "Please expand your filter criteria or click 'Reset all filters' in the sidebar."
    )
    st.stop()

data_hash = dataset.sha256 if hasattr(dataset, "sha256") else "default"
results = run_analytics_cached(data_hash, filter_state.to_dict())

findings = results["findings"]
kpis_v2 = results["kpis_v2"]
is_fast_mode = not filter_state.is_default(dataset)

# ── 4. Executive Header ───────────────────────────────────────────────────────
n_t = view.counts.get("n_teachers_active", view.counts.get("n_teachers", 0))
n_c = view.counts.get("n_courses_active", view.counts.get("n_courses", 0))
n_p = view.counts.get("n_pairs", 0)
n_e = view.counts.get("n_enrollments", 0)

st.markdown(
    """
<div style='margin-bottom: 0.5rem;'>
    <div style='font-size: 1.55rem; font-weight: 800; color: #0F172A; letter-spacing: -0.01em;'>
        EduPro Analytics Platform
    </div>
    <div style='font-size: 0.85rem; color: #475569; margin-top: 0.15rem;'>
        Enterprise Teaching Performance & Course Quality Evaluation
    </div>
</div>
""",
    unsafe_allow_html=True,
)

fast_badge = " · Fast estimation mode active (300 bootstrap iterations)" if is_fast_mode else ""
active_filters_str = (
    f" · Active filters: {', '.join(view.active_filters)}" if view.active_filters else ""
)

st.caption(
    f"Evaluated Scope: {n_t:,} instructors, {n_c:,} courses "
    f"({n_p:,} transaction pairs, {n_e:,} enrollments){fast_badge}{active_filters_str}"
)

# ── 5. Platform Key Performance Indicators ────────────────────────────────────
st.markdown("##### Platform Performance Indicators")

k1, k2, k3, k4, k5 = st.columns(5)

kpi_tile(
    k1,
    label="Avg Teacher Rating",
    kpi=kpis_v2["avg_teacher_rating"],
    format_spec=".3f",
    unit_suffix=" / 5.0",
)

kpi_tile(
    k2,
    label="Avg Course Rating",
    kpi=kpis_v2["avg_course_rating"],
    format_spec=".4f",
    unit_suffix=" / 5.0",
)

kpi_tile(
    k3,
    label="Quality Consistency",
    kpi=kpis_v2["rating_consistency_index"],
    format_spec=".3f",
)

exp_kpi = kpis_v2["experience_impact"]
slope_tr = exp_kpi.extra.get("slope_on_teacher_rating", float("nan"))
slope_cr = exp_kpi.extra.get("slope_on_course_rating", float("nan"))
ci_tr = exp_kpi.extra.get("slope_on_teacher_rating_ci", (exp_kpi.ci_low, exp_kpi.ci_high))
ci_cr = exp_kpi.extra.get("slope_on_course_rating_ci", (float("nan"), float("nan")))


def _fmt_slope_val(val: float) -> str:
    if math.isnan(val):
        return "N/A"
    s = f"{val:+.3f}"
    return f"{val:+.4f}" if s in ("+0.000", "-0.000") else s


def _fmt_ci_range(ci: tuple[float, float]) -> str:
    if math.isnan(ci[0]) or math.isnan(ci[1]):
        return "N/A"
    return f"[{_fmt_slope_val(ci[0])}, {_fmt_slope_val(ci[1])}]"


delta_cr_str = (
    f"{_fmt_slope_val(slope_cr)} per year on courses (95% CI: {_fmt_ci_range(ci_cr)})"
    if not math.isnan(slope_cr)
    else None
)
subcaption_str = f"n = {exp_kpi.n} | TR CI: {_fmt_ci_range(ci_tr)} | CR CI: {_fmt_ci_range(ci_cr)}"

kpi_tile(
    k4,
    label="Experience Impact",
    kpi=exp_kpi,
    delta=delta_cr_str,
    delta_color="off",
    format_spec="+.3f",
    unit_suffix=" per year",
    subcaption=subcaption_str,
)

kpi_tile(
    k5,
    label="Enrollment Lift (Top Instructors)",
    kpi=kpis_v2["enrollment_influence"],
    format_spec=".2f",
    unit_suffix="×",
)

st.markdown("---")

# ── 6. Analytical Tabs ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Instructor Rankings",
        "Experience & Ratings",
        "Course Quality Maps",
        "Domain Analysis",
        "Rating Relationships",
    ]
)

with tab1:
    tab1_rankings.render(view, findings, kpis_v2, is_fast_mode=is_fast_mode)

with tab2:
    tab2_experience.render(view, findings, kpis_v2, is_fast_mode=is_fast_mode)

with tab3:
    tab3_course_quality.render(view, findings, kpis_v2, is_fast_mode=is_fast_mode)

with tab4:
    tab4_domain.render(view, findings, kpis_v2, is_fast_mode=is_fast_mode)

with tab5:
    tab5_relationships.render(view, findings, kpis_v2, is_fast_mode=is_fast_mode)
