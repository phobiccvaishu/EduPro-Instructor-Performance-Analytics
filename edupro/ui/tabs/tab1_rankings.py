"""
edupro.ui.tabs.tab1_rankings
============================
Tab 1 — Instructor Performance, Quality Rankings, and Enrollment Distribution.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from edupro.insights import generate_f1_insight, generate_f10_f11_insight
from edupro.ui.charts import build_horizontal_bar
from edupro.ui.components import (
    empty_state,
    insight_box,
    method_note,
    scope_caption,
    section_header,
)
from edupro.ui.theme import TIER_COLORS


def render(
    view: Any, findings: dict[str, Any], kpis: dict[str, Any], is_fast_mode: bool = False
) -> None:
    """Render Tab 1 view."""
    section_header(
        "Instructor Performance & Quality Rankings",
        "Evaluation of instructor teaching ratings, tier distribution, and enrollment concentration.",
    )
    scope_caption(view, unit="instructor", is_fast_mode=is_fast_mode)

    if view.is_empty or len(view.teacher_table) == 0:
        empty_state("No instructor records match the selected filter combination.")
        return

    teachers_df = view.teacher_table.copy()

    # 1. Generated Dynamic Insights
    if "F1" in findings:
        f1_ins = generate_f1_insight(findings["F1"], is_fast_mode=is_fast_mode)
        insight_box(
            headline=f1_ins.headline,
            evidence_text=f1_ins.evidence_sentence,
            caveat_text=f1_ins.caveat,
            verdict=f1_ins.verdict,
            evidence_level=f1_ins.evidence_level,
            stat_meta=f1_ins.stat_summary,
        )

    if "F10" in findings and "F11" in findings:
        f11_ins = generate_f10_f11_insight(
            findings["F10"], findings["F11"], is_fast_mode=is_fast_mode
        )
        insight_box(
            headline=f11_ins.headline,
            evidence_text=f11_ins.evidence_sentence,
            caveat_text=f11_ins.caveat,
            verdict=f11_ins.verdict,
            evidence_level=f11_ins.evidence_level,
            stat_meta=f11_ins.stat_summary,
        )

    # 2. Methodology & Governance Note
    method_note(
        "Instructor Scoring & Tier Attribution",
        [
            ("Unit of Analysis", "Deduplicated instructor (n = 60 baseline)."),
            (
                "Statistical Tests",
                "Five-number summary, Shapiro-Wilk normality test, Pareto concentration with bootstrap CIs.",
            ),
            (
                "Attribution Rule",
                "Course quality scores linked via transactions are weighted by student enrollment volume.",
            ),
            (
                "Referential Scope",
                "Instructors with 0 enrollments in the active course scope are dropped from entity tables.",
            ),
        ],
    )

    # 3. Main Ranking Dataframe
    st.markdown("##### Complete Instructor League Table")
    rank_df = teachers_df.sort_values("TeacherRating", ascending=False).reset_index(drop=True)
    rank_df["Rank"] = rank_df.index + 1

    display_cols = [
        "Rank",
        "TeacherName",
        "Expertise",
        "YearsOfExperience",
        "TeacherRating",
        "InstructorTier",
        "total_enrollments",
        "weighted_course_rating",
        "n_courses",
    ]
    available_cols = [c for c in display_cols if c in rank_df.columns]

    st.dataframe(
        rank_df[available_cols],
        width="stretch",
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "TeacherName": st.column_config.TextColumn("Instructor", width="medium"),
            "Expertise": st.column_config.TextColumn("Domain", width="medium"),
            "YearsOfExperience": st.column_config.NumberColumn(
                "Experience (Yrs)", format="%d yrs", width="small"
            ),
            "TeacherRating": st.column_config.NumberColumn(
                "Teacher Rating", format="%.2f / 5.0", width="medium"
            ),
            "InstructorTier": st.column_config.TextColumn("Quality Tier", width="small"),
            "total_enrollments": st.column_config.NumberColumn(
                "Enrollments", format="%d", width="medium"
            ),
            "weighted_course_rating": st.column_config.NumberColumn(
                "Weighted Course Score", format="%.2f / 5.0", width="medium"
            ),
            "n_courses": st.column_config.NumberColumn(
                "Courses Taught", format="%d", width="small"
            ),
        },
    )

    st.markdown("---")

    # 4. Tier Distribution Summary Table
    t1, t2 = st.columns([1, 1], gap="large")

    with t1:
        st.markdown("##### Performance by Quality Tier")
        if "InstructorTier" in teachers_df.columns:
            tier_grp = (
                teachers_df.groupby("InstructorTier", observed=False)
                .agg(
                    Instructors=("TeacherID", "count"),
                    Total_Enrollments=("total_enrollments", "sum"),
                    Avg_Teacher_Rating=("TeacherRating", "mean"),
                    Avg_Course_Rating=("weighted_course_rating", "mean"),
                )
                .reset_index()
            )

            tot_inst = len(teachers_df)
            tot_enr = teachers_df["total_enrollments"].sum()
            tier_grp["Instructor_Share"] = (tier_grp["Instructors"] / tot_inst * 100.0).round(1)
            tier_grp["Enrollment_Share"] = (tier_grp["Total_Enrollments"] / tot_enr * 100.0).round(
                1
            )

            st.dataframe(
                tier_grp[
                    [
                        "InstructorTier",
                        "Instructors",
                        "Instructor_Share",
                        "Total_Enrollments",
                        "Enrollment_Share",
                        "Avg_Teacher_Rating",
                        "Avg_Course_Rating",
                    ]
                ],
                width="stretch",
                hide_index=True,
                column_config={
                    "InstructorTier": st.column_config.TextColumn("Tier"),
                    "Instructors": st.column_config.NumberColumn("Count", format="%d"),
                    "Instructor_Share": st.column_config.NumberColumn(
                        "% Instructors", format="%.1f%%"
                    ),
                    "Total_Enrollments": st.column_config.NumberColumn("Enrollments", format="%d"),
                    "Enrollment_Share": st.column_config.NumberColumn(
                        "% Enrollments", format="%.1f%%"
                    ),
                    "Avg_Teacher_Rating": st.column_config.NumberColumn(
                        "Avg Instructor Score", format="%.2f"
                    ),
                    "Avg_Course_Rating": st.column_config.NumberColumn(
                        "Avg Course Score", format="%.2f"
                    ),
                },
            )

    with t2:
        st.markdown("##### High vs Low Cohort Metrics")
        high_df = teachers_df[teachers_df["TeacherRating"] >= 4.0]
        low_df = teachers_df[teachers_df["TeacherRating"] < 3.0]
        n_high = len(high_df)
        n_low = len(low_df)
        high_enr = high_df["total_enrollments"].mean() if n_high > 0 else 0.0
        low_enr = low_df["total_enrollments"].mean() if n_low > 0 else 0.0

        st.markdown(
            f"""
<div style='background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.85rem; border-radius: 4px; font-size: 0.86rem;'>
    <div style='margin-bottom: 0.5rem;'><strong>High-Rated Instructors (>= 4.0):</strong> {n_high} instructors ({n_high / max(tot_inst, 1) * 100:.1f}%)</div>
    <div style='margin-bottom: 0.5rem;'><strong>Mean Enrollments per High-Tier Instructor:</strong> {high_enr:.0f} students</div>
    <div style='margin-bottom: 0.5rem;'><strong>Low-Rated Instructors (&lt; 3.0):</strong> {n_low} instructors ({n_low / max(tot_inst, 1) * 100:.1f}%)</div>
    <div style='margin-bottom: 0.5rem;'><strong>Mean Enrollments per Low-Tier Instructor:</strong> {low_enr:.0f} students</div>
    <div style='color: #64748B; font-size: 0.80rem; margin-top: 0.4rem;'>Data indicates substantial enrollment lift for top-tier instructors without direct translation to course content quality.</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 5. Top 15 and Bottom 10 Visualizations
    c1, c2 = st.columns(2, gap="large")

    with c1:
        top15 = teachers_df.sort_values("TeacherRating", ascending=True).tail(15)
        fig_top = build_horizontal_bar(
            top15,
            x_col="TeacherRating",
            y_col="TeacherName",
            title="Top 15 Instructors by Teacher Rating",
            x_title="Teacher Rating (1–5 scale)",
            y_title="Instructor",
            color_col="InstructorTier" if "InstructorTier" in top15.columns else None,
            color_map=TIER_COLORS,
            height=420,
            x_range=[0.0, 5.5],
        )
        st.plotly_chart(fig_top, width="stretch")

    with c2:
        bot10 = teachers_df.sort_values("TeacherRating", ascending=False).tail(10)
        fig_bot = build_horizontal_bar(
            bot10,
            x_col="TeacherRating",
            y_col="TeacherName",
            title="Bottom 10 Instructors (Support & Coaching Priority)",
            x_title="Teacher Rating (1–5 scale)",
            y_title="Instructor",
            color_col="InstructorTier" if "InstructorTier" in bot10.columns else None,
            color_map=TIER_COLORS,
            height=420,
            x_range=[0.0, 5.5],
        )
        st.plotly_chart(fig_bot, width="stretch")
