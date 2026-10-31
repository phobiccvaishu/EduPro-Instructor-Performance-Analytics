"""
edupro.ui.tabs.tab2_experience
==============================
Tab 2 — Teaching Experience vs Instructor and Course Quality Ratings.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from edupro.analytics.kpis import format_slope
from edupro.formatting import fmt_p_value
from edupro.insights import generate_f2_insight, generate_f3_insight, generate_f4_insight
from edupro.ui.charts import build_scatter_with_trendline
from edupro.ui.components import (
    empty_state,
    insight_box,
    method_note,
    scope_caption,
    section_header,
)


def render(
    view: Any, findings: dict[str, Any], kpis: dict[str, Any], is_fast_mode: bool = False
) -> None:
    """Render Tab 2 view."""
    section_header(
        "Teaching Experience & Performance Evaluation",
        "Investigating empirical relationships between years of experience, teacher ratings, and course quality.",
    )
    scope_caption(view, unit="instructor & transaction pair", is_fast_mode=is_fast_mode)

    if view.is_empty or len(view.teacher_table) == 0:
        empty_state("No data available for the active filter scope.")
        return

    teachers_df = view.teacher_table.copy()
    master_df = view.master.copy()

    # 1. Generated Dynamic Insights
    if "F2" in findings:
        f2_ins = generate_f2_insight(findings["F2"], is_fast_mode=is_fast_mode)
        insight_box(
            headline=f2_ins.headline,
            evidence_text=f2_ins.evidence_sentence,
            caveat_text=f2_ins.caveat,
            verdict=f2_ins.verdict,
            evidence_level=f2_ins.evidence_level,
            stat_meta=f2_ins.stat_summary,
        )

    if "F3" in findings:
        f3_ins = generate_f3_insight(findings["F3"], is_fast_mode=is_fast_mode)
        insight_box(
            headline=f3_ins.headline,
            evidence_text=f3_ins.evidence_sentence,
            caveat_text=f3_ins.caveat,
            verdict=f3_ins.verdict,
            evidence_level=f3_ins.evidence_level,
            stat_meta=f3_ins.stat_summary,
        )

    if "F4a" in findings and "F4b" in findings:
        f4_ins = generate_f4_insight(findings["F4a"], findings["F4b"], is_fast_mode=is_fast_mode)
        insight_box(
            headline=f4_ins.headline,
            evidence_text=f4_ins.evidence_sentence,
            caveat_text=f4_ins.caveat,
            verdict=f4_ins.verdict,
            evidence_level=f4_ins.evidence_level,
            stat_meta=f4_ins.stat_summary,
        )

    # 2. Methodology & Governance Note
    method_note(
        "Experience Modeling & Statistical Controls",
        [
            (
                "Observational Units",
                "Teacher scores modeled at instructor level (n = 60); course quality at transaction level with clustered errors.",
            ),
            (
                "Statistical Controls",
                "HC3 heteroskedasticity-consistent standard errors; piecewise regression with permutation test.",
            ),
            (
                "Decoupling Evidence",
                "Experience predicts instructor score positively while showing negligible translation into course content ratings.",
            ),
            (
                "Small Sample Policy",
                "Trendlines are suppressed when active cohort contains fewer than 3 records or uniform experience.",
            ),
        ],
    )

    # 3. High-level metric summary cards
    m1, m2, m3, m4 = st.columns(4)
    f2_obj = findings.get("F2")
    f3_obj = findings.get("F3")
    f4a_obj = findings.get("F4a")
    f4b_obj = findings.get("F4b")

    slope_tr = f2_obj.extra.get("ols_slope", float("nan")) if f2_obj else float("nan")
    r_tr = f2_obj.statistic if f2_obj else float("nan")
    p_tr = f2_obj.p_value if f2_obj else float("nan")

    slope_cr = f3_obj.extra.get("ols_slope", float("nan")) if f3_obj else float("nan")
    r_cr = f3_obj.statistic if f3_obj else float("nan")
    p_cr = f3_obj.p_value if f3_obj else float("nan")

    bp_val = f4b_obj.statistic if f4b_obj else float("nan")
    diff_val = f4a_obj.statistic if f4a_obj else float("nan")

    tr_metric_val = (
        f"{format_slope(slope_tr, signed=True)}/yr" if not math.isnan(slope_tr) else "N/A"
    )
    cr_metric_val = (
        f"{format_slope(slope_cr, signed=True)}/yr" if not math.isnan(slope_cr) else "N/A"
    )
    bp_metric_val = f"{bp_val:.0f} Years" if not math.isnan(bp_val) else "N/A"
    diff_metric_val = f"+{diff_val:.2f} Pts" if not math.isnan(diff_val) else "N/A"

    m1.metric(
        "Experience -> Teacher Rating",
        tr_metric_val,
        help=f"Pearson r = {r_tr:.3f}, {fmt_p_value(p_tr)}",
    )
    m2.metric(
        "Experience -> Course Quality",
        cr_metric_val,
        help=f"Pearson r = {r_cr:.3f}, {fmt_p_value(p_cr)} (Not statistically significant)",
    )
    m3.metric(
        "Inflection Breakpoint",
        bp_metric_val,
        help="Piecewise regression inflection threshold separating performance tiers.",
    )
    m4.metric(
        "Senior vs Early Career Lift",
        diff_metric_val,
        help="Mean rating difference between instructors with 6+ years experience vs 0-5 years.",
    )

    st.markdown("---")

    # 4. Scatter Charts: Experience vs Teacher Rating & Experience vs Course Rating
    c1, c2 = st.columns(2, gap="large")

    f2_slope_ci = f2_obj.extra.get("ols_slope_ci") if f2_obj else None
    f3_slope_ci = f3_obj.extra.get("pair_level_cluster_robust", {}).get("ci") if f3_obj else None

    with c1:
        fig_s1 = build_scatter_with_trendline(
            teachers_df,
            x_col="YearsOfExperience",
            y_col="TeacherRating",
            title="Years of Experience vs Instructor Rating",
            x_title="Years of Teaching Experience",
            y_title="Instructor Rating (1–5 scale)",
            slope=slope_tr,
            slope_ci=f2_slope_ci,
            hover_name="TeacherName",
            height=400,
            y_range=[0.5, 5.5],
        )
        st.plotly_chart(fig_s1, width="stretch")

    with c2:
        fig_s2 = build_scatter_with_trendline(
            master_df,
            x_col="YearsOfExperience",
            y_col="CourseRating",
            title="Years of Experience vs Course Rating",
            x_title="Years of Teaching Experience",
            y_title="Course Rating (1–5 scale)",
            slope=slope_cr,
            slope_ci=f3_slope_ci,
            hover_name="CourseName" if "CourseName" in master_df.columns else None,
            height=400,
            y_range=[0.5, 5.5],
        )
        st.plotly_chart(fig_s2, width="stretch")

    st.markdown("---")

    # 5. Experience Brackets Summary Table
    st.markdown("##### Performance Breakdown by Teaching Experience Bracket")
    if "YearsOfExperience" in teachers_df.columns:
        bins = [-0.1, 5.0, 10.0, 15.0, 100.0]
        labels = ["0-5 yrs", "6-10 yrs", "11-15 yrs", "16+ yrs"]
        teachers_df["Exp_Bracket"] = pd.cut(
            teachers_df["YearsOfExperience"], bins=bins, labels=labels
        )

        bracket_summary = (
            teachers_df.groupby("Exp_Bracket", observed=False)
            .agg(
                Instructors=("TeacherID", "count"),
                Avg_Teacher_Rating=("TeacherRating", "mean"),
                Min_Teacher_Rating=("TeacherRating", "min"),
                Max_Teacher_Rating=("TeacherRating", "max"),
                Avg_Course_Rating=("weighted_course_rating", "mean"),
                Total_Enrollments=("total_enrollments", "sum"),
            )
            .reset_index()
        )

        st.dataframe(
            bracket_summary,
            width="stretch",
            hide_index=True,
            column_config={
                "Exp_Bracket": st.column_config.TextColumn("Experience Bracket"),
                "Instructors": st.column_config.NumberColumn("Instructors", format="%d"),
                "Avg_Teacher_Rating": st.column_config.NumberColumn(
                    "Avg Instructor Score", format="%.2f"
                ),
                "Min_Teacher_Rating": st.column_config.NumberColumn("Min Score", format="%.2f"),
                "Max_Teacher_Rating": st.column_config.NumberColumn("Max Score", format="%.2f"),
                "Avg_Course_Rating": st.column_config.NumberColumn(
                    "Avg Course Quality", format="%.2f"
                ),
                "Total_Enrollments": st.column_config.NumberColumn(
                    "Total Enrollments", format="%d"
                ),
            },
        )
