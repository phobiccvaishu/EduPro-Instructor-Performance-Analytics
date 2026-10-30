"""
edupro.ui.tabs.tab5_relationships
=================================
Tab 5 — Rating Relationships, Distributions, and Cluster-Aware Correlation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from edupro.insights import generate_f5_insight
from edupro.ui.charts import (
    build_distribution_histogram,
    build_heatmap_matrix,
    build_scatter_with_trendline,
)
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
    """Render Tab 5 view."""
    section_header(
        "Rating Relationships & Statistical Decoupling",
        "Evaluating joint distributions and testing correlation between instructor scores and course quality.",
    )
    scope_caption(view, unit="transaction pair & instructor", is_fast_mode=is_fast_mode)

    if view.is_empty or len(view.master) == 0:
        empty_state("No transaction data available for the active filter selection.")
        return

    teachers_df = view.teacher_table.copy()
    courses_df = view.course_table.copy()
    master_df = view.master.copy()

    # 1. Generated Dynamic Insight
    if "F5" in findings:
        f5_ins = generate_f5_insight(findings["F5"], is_fast_mode=is_fast_mode)
        insight_box(
            headline=f5_ins.headline,
            evidence_text=f5_ins.evidence_sentence,
            caveat_text=f5_ins.caveat,
            verdict=f5_ins.verdict,
            evidence_level=f5_ins.evidence_level,
            stat_meta=f5_ins.stat_summary,
        )

    # 2. Methodology & Governance Note
    method_note(
        "Cluster-Aware Inference & Distribution Invariants",
        [
            (
                "Observational Pairs",
                f"Transaction pair level (n = {len(master_df):,} active pairs across {len(teachers_df)} instructors and {len(courses_df)} courses).",
            ),
            (
                "Pseudo-Replication Guard",
                "Instructors teach multiple courses and courses are taught by multiple instructors. Conventional OLS yields artificially deflated p-values.",
            ),
            (
                "Clustering Method",
                "Inference employs cluster-robust sandwich covariance estimators clustering on TeacherID.",
            ),
        ],
    )

    # 3. Scatter Plot: Teacher Rating vs Course Rating with cluster-aware regression slope
    f5_obj = findings.get("F5")
    slope_tc = f5_obj.statistic if f5_obj else 0.0
    slope_ci = [f5_obj.ci_low, f5_obj.ci_high] if f5_obj else None

    fig_pair_scatter = build_scatter_with_trendline(
        master_df,
        x_col="TeacherRating",
        y_col="CourseRating",
        title="Instructor Rating vs Course Content Rating (Transaction Pairs)",
        x_title="Instructor Teaching Score (1–5 scale)",
        y_title="Course Content Score (1–5 scale)",
        slope=slope_tc,
        slope_ci=slope_ci,
        hover_name="CourseName" if "CourseName" in master_df.columns else None,
        height=420,
        y_range=[0.5, 5.5],
    )
    st.plotly_chart(fig_pair_scatter, width="stretch")

    st.markdown("---")

    # 4. Marginal Distributions: Teacher Rating and Course Rating Histograms
    d1, d2 = st.columns(2, gap="large")

    with d1:
        fig_th = build_distribution_histogram(
            teachers_df,
            col_name="TeacherRating",
            title="Instructor Rating Distribution (Deduplicated)",
            x_title="Instructor Rating (1–5 scale)",
            nbins=15,
            height=340,
        )
        st.plotly_chart(fig_th, width="stretch")

    with d2:
        fig_ch = build_distribution_histogram(
            courses_df,
            col_name="CourseRating",
            title="Course Catalog Rating Distribution",
            x_title="Course Rating (1–5 scale)",
            nbins=15,
            height=340,
        )
        st.plotly_chart(fig_ch, width="stretch")

    st.markdown("---")

    # 5. Quality Tier Cross-Tabulation Matrix
    st.markdown("##### Performance Cohort Cross-Tabulation (Teacher Tier vs Course Tier)")

    # Define bins for tiers
    tier_bins = [0.0, 3.0, 4.0, 5.0]
    tier_labels = ["Low (<3.0)", "Mid (3.0-3.9)", "High (>=4.0)"]

    master_df["Teacher_Cohort"] = pd.cut(
        master_df["TeacherRating"], bins=tier_bins, labels=tier_labels
    )
    master_df["Course_Cohort"] = pd.cut(
        master_df["CourseRating"], bins=tier_bins, labels=tier_labels
    )

    cohort_matrix = master_df.pivot_table(
        index="Teacher_Cohort",
        columns="Course_Cohort",
        values="TransactionID" if "TransactionID" in master_df.columns else "EnrollmentCount",
        aggfunc="count",
        observed=False,
    ).fillna(0)

    fig_matrix = build_heatmap_matrix(
        cohort_matrix,
        title="Transaction Pair Distribution Across Quality Tiers",
        x_title="Course Quality Cohort",
        y_title="Instructor Quality Cohort",
        text_format=".0f",
        height=380,
    )
    st.plotly_chart(fig_matrix, width="stretch")
