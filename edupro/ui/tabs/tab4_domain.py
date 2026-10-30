"""
edupro.ui.tabs.tab4_domain
==========================
Tab 4 — Domain Analysis, Subject Expertise Contrasts, and ML Gap Evaluation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from edupro.insights import generate_f8_insight, generate_f13_insight
from edupro.ui.charts import apply_base_layout
from edupro.ui.components import (
    empty_state,
    insight_box,
    method_note,
    scope_caption,
    section_header,
)
from edupro.ui.theme import COLOR_SERIES_PRIMARY, COLOR_SERIES_SECONDARY


def render(
    view: Any, findings: dict[str, Any], kpis: dict[str, Any], is_fast_mode: bool = False
) -> None:
    """Render Tab 4 view."""
    section_header(
        "Domain Performance & Quality Discrepancies",
        "Analyzing teaching performance across subject domains and contrasting instructor ratings against course quality.",
    )
    scope_caption(view, unit="expertise domain & instructor", is_fast_mode=is_fast_mode)

    if view.is_empty or len(view.teacher_table) == 0:
        empty_state("No domain data available for the active filter selection.")
        return

    teachers_df = view.teacher_table.copy()
    master_df = view.master.copy()

    # 1. Generated Dynamic Insights
    if "F8a" in findings and "F8b" in findings and "F8c" in findings:
        f8_ins = generate_f8_insight(
            findings["F8a"], findings["F8b"], findings["F8c"], is_fast_mode=is_fast_mode
        )
        insight_box(
            headline=f8_ins.headline,
            evidence_text=f8_ins.evidence_sentence,
            caveat_text=f8_ins.caveat,
            verdict=f8_ins.verdict,
            evidence_level=f8_ins.evidence_level,
            stat_meta=f8_ins.stat_summary,
        )

    if "F13" in findings:
        f13_ins = generate_f13_insight(findings["F13"], is_fast_mode=is_fast_mode)
        insight_box(
            headline=f13_ins.headline,
            evidence_text=f13_ins.evidence_sentence,
            caveat_text=f13_ins.caveat,
            verdict=f13_ins.verdict,
            evidence_level=f13_ins.evidence_level,
            stat_meta=f13_ins.stat_summary,
        )

    # 2. Methodology & Governance Note
    method_note(
        "Domain Contrasts & Family-Wise Error Control",
        [
            (
                "Observational Units",
                "Instructor score aggregated across deduplicated teachers; course score averaged across transaction links.",
            ),
            (
                "Post-Hoc Adjustments",
                "All pairwise domain comparisons control family-wise error rate via Holm-Bonferroni step-down adjustments.",
            ),
            (
                "Conflation Warning",
                "Machine Learning instructors exhibit high scores (4.23), yet ML courses average 2.66; this reflects content design rather than instructor deficit.",
            ),
        ],
    )

    # 3. Domain Comparison Aggregation Table
    inst_by_exp = (
        teachers_df.groupby("Expertise", observed=False)
        .agg(
            Instructors=("TeacherID", "count"),
            Avg_Teacher_Rating=("TeacherRating", "mean"),
            Total_Enrollments=("total_enrollments", "sum"),
        )
        .reset_index()
    )

    course_by_exp = (
        master_df.groupby("Expertise", observed=False)
        .agg(
            Avg_Course_Rating=("CourseRating", "mean"),
        )
        .reset_index()
    )

    domain_merged = pd.merge(inst_by_exp, course_by_exp, on="Expertise").sort_values(
        "Avg_Teacher_Rating", ascending=False
    )
    domain_merged["Rating_Gap"] = (
        domain_merged["Avg_Teacher_Rating"] - domain_merged["Avg_Course_Rating"]
    ).round(3)

    st.markdown("##### Domain Performance & Quality Gap Summary")
    st.dataframe(
        domain_merged[
            [
                "Expertise",
                "Instructors",
                "Avg_Teacher_Rating",
                "Avg_Course_Rating",
                "Rating_Gap",
                "Total_Enrollments",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "Expertise": st.column_config.TextColumn("Teaching Domain"),
            "Instructors": st.column_config.NumberColumn("Instructors", format="%d"),
            "Avg_Teacher_Rating": st.column_config.NumberColumn(
                "Avg Instructor Score", format="%.2f"
            ),
            "Avg_Course_Rating": st.column_config.NumberColumn("Avg Course Score", format="%.2f"),
            "Rating_Gap": st.column_config.NumberColumn("Gap (Teacher - Course)", format="%+.2f"),
            "Total_Enrollments": st.column_config.NumberColumn("Total Enrollments", format="%d"),
        },
    )

    st.markdown("---")

    # 4. Side-by-side or paired bar chart: Instructor Score vs Course Score
    c1, c2 = st.columns(2, gap="large")

    with c1:
        fig_bar = go.Figure()
        sorted_domains = domain_merged.sort_values("Avg_Teacher_Rating", ascending=True)

        fig_bar.add_trace(
            go.Bar(
                y=sorted_domains["Expertise"],
                x=sorted_domains["Avg_Teacher_Rating"],
                name="Instructor Rating",
                orientation="h",
                marker_color=COLOR_SERIES_PRIMARY,
                text=sorted_domains["Avg_Teacher_Rating"].apply(lambda v: f"{v:.2f}"),
                textposition="outside",
            )
        )
        fig_bar.add_trace(
            go.Bar(
                y=sorted_domains["Expertise"],
                x=sorted_domains["Avg_Course_Rating"],
                name="Course Rating",
                orientation="h",
                marker_color=COLOR_SERIES_SECONDARY,
                text=sorted_domains["Avg_Course_Rating"].apply(lambda v: f"{v:.2f}"),
                textposition="outside",
            )
        )

        fig_bar.update_layout(
            barmode="group",
            xaxis=dict(title="Rating (1–5 scale)", range=[0.0, 5.5]),
            yaxis=dict(title="Domain"),
            legend=dict(orientation="h", y=-0.2, x=0.0),
        )
        apply_base_layout(fig_bar, "Instructor Rating vs Course Rating by Domain", height=460)
        st.plotly_chart(fig_bar, width="stretch")

    with c2:
        # Bubble Scatter: Instructor Score vs Course Score (Bubble size = enrollments)
        fig_bubble = px.scatter(
            domain_merged,
            x="Avg_Teacher_Rating",
            y="Avg_Course_Rating",
            size="Total_Enrollments",
            color="Rating_Gap",
            color_continuous_scale="RdBu_r",
            text="Expertise",
            hover_data={
                "Expertise": True,
                "Avg_Teacher_Rating": ":.2f",
                "Avg_Course_Rating": ":.2f",
                "Rating_Gap": ":+.2f",
                "Total_Enrollments": True,
            },
            labels={
                "Avg_Teacher_Rating": "Average Instructor Rating",
                "Avg_Course_Rating": "Average Course Rating",
                "Rating_Gap": "Quality Gap",
            },
        )
        fig_bubble.update_traces(
            textposition="top center",
            textfont=dict(size=10, color="#0F172A"),
        )
        fig_bubble.update_layout(
            xaxis=dict(range=[2.0, 4.8]),
            yaxis=dict(range=[2.2, 4.0]),
            coloraxis_colorbar=dict(title="Gap", thickness=12, len=0.8),
        )
        apply_base_layout(
            fig_bubble, "Domain Rating Matrix (Bubble Size = Enrollments)", height=460
        )
        st.plotly_chart(fig_bubble, width="stretch")
