"""
edupro.analytics.engine
=======================
Six-stage modular analytical methodology engine for EduPro platform.
Provides complete analysis across Integration, Profiles, Experience, Quality, Impact, and Expertise.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from edupro.analytics.kpis import KPIEngine
from edupro.config import settings
from edupro.data.validation import validate_mapping
from edupro.logging_setup import get_logger

logger = get_logger("analytics.engine")


class AnalyticsEngine:
    """
    Six-stage modular analytical engine.

    Parameters
    ----------
    master : pd.DataFrame
        Integrated master DataFrame.
    teachers_df : pd.DataFrame
        Cleaned Teachers DataFrame.
    courses_df : pd.DataFrame
        Cleaned Courses DataFrame.
    transactions_df : pd.DataFrame
        Cleaned Transactions DataFrame.
    """

    def __init__(
        self,
        master: pd.DataFrame,
        teachers_df: pd.DataFrame,
        courses_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
    ) -> None:
        self.master = master.copy()
        self.teachers = teachers_df.copy()
        self.courses = courses_df.copy()
        self.txns = transactions_df.copy()
        self.kpi = KPIEngine(master)

    # ------------------------------------------------------------------
    # Stage 1 — Data Integration
    # ------------------------------------------------------------------
    def data_integration_summary(self) -> dict[str, Any]:
        """
        Summarise integration: unique IDs, join stats, mapping accuracy.
        """
        report = validate_mapping(self.master)
        report["teacher_ids_in_transactions"] = self.txns["TeacherID"].nunique()
        report["course_ids_in_transactions"] = self.txns["CourseID"].nunique()
        return report

    # ------------------------------------------------------------------
    # Stage 2 — Instructor Profile Analysis
    # ------------------------------------------------------------------
    def instructor_profiles(self) -> dict[str, Any]:
        """
        Distributions of Age, YearsOfExperience, Expertise;
        rating spread; top/bottom performers.
        """
        t = self.teachers.copy()

        rating_stats = t["TeacherRating"].describe().round(3).to_dict()

        # Top 10 performers
        top10 = (
            t.sort_values("TeacherRating", ascending=False)
            .head(10)[
                [
                    "TeacherID",
                    "TeacherName",
                    "Expertise",
                    "YearsOfExperience",
                    "TeacherRating",
                ]
            ]
            .reset_index(drop=True)
        )

        # Bottom 10 performers
        bottom10 = (
            t.sort_values("TeacherRating", ascending=True)
            .head(10)[
                [
                    "TeacherID",
                    "TeacherName",
                    "Expertise",
                    "YearsOfExperience",
                    "TeacherRating",
                ]
            ]
            .reset_index(drop=True)
        )

        # Expertise distribution
        expertise_dist = (
            t.groupby("Expertise")
            .agg(
                count=("TeacherID", "count"),
                avg_rating=("TeacherRating", "mean"),
                avg_exp=("YearsOfExperience", "mean"),
            )
            .round(3)
            .reset_index()
            .sort_values("avg_rating", ascending=False)
        )

        return {
            "rating_stats": rating_stats,
            "top10": top10,
            "bottom10": bottom10,
            "expertise_dist": expertise_dist,
        }

    # ------------------------------------------------------------------
    # Stage 3 — Experience vs. Performance Analysis
    # ------------------------------------------------------------------
    def experience_vs_performance(self) -> dict[str, Any]:
        """
        Correlations between YearsOfExperience and ratings;
        experience threshold / diminishing returns analysis.
        """
        t = self.teachers.dropna(subset=["YearsOfExperience", "TeacherRating"])

        r_t, p_t = stats.pearsonr(t["YearsOfExperience"], t["TeacherRating"])
        r_s, p_s = stats.spearmanr(t["YearsOfExperience"], t["TeacherRating"])

        # Quartile-based binning
        t = t.copy()
        t["exp_bin"] = pd.qcut(
            t["YearsOfExperience"],
            q=4,
            labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"],
            duplicates="drop",
        )
        quartile_avg = (
            t.groupby("exp_bin", observed=True)["TeacherRating"]
            .mean()
            .round(3)
            .reset_index()
        )

        # Experience bins (0-5, 6-10, 11-15, 16+)
        bins = list(settings.exp_bins)
        labels = list(settings.exp_labels)
        t["exp_group"] = pd.cut(
            t["YearsOfExperience"], bins=bins, labels=labels, right=True
        )
        group_avg = (
            t.groupby("exp_group", observed=True)["TeacherRating"]
            .agg(["mean", "count"])
            .round(3)
            .reset_index()
        )

        # Course-level correlation (using master)
        mc = self.master.dropna(subset=["YearsOfExperience", "CourseRating"])
        r_c, p_c = stats.pearsonr(mc["YearsOfExperience"], mc["CourseRating"])

        return {
            "pearson_r_teacher": round(float(r_t), 4),
            "pearson_p_teacher": round(float(p_t), 6),
            "spearman_r_teacher": round(float(r_s), 4),
            "spearman_p_teacher": round(float(p_s), 6),
            "pearson_r_course": round(float(r_c), 4),
            "pearson_p_course": round(float(p_c), 6),
            "quartile_avg": quartile_avg,
            "group_avg": group_avg,
        }

    # ------------------------------------------------------------------
    # Stage 4 — Course Quality Evaluation
    # ------------------------------------------------------------------
    def course_quality_evaluation(self) -> dict[str, Any]:
        """
        CourseRating by Category and Level; Gender x Level comparison.
        """
        courses_m = self.master.drop_duplicates(subset="CourseID").copy()

        # By Category
        by_category = (
            courses_m.groupby("CourseCategory")["CourseRating"]
            .agg(avg_rating="mean", count="count", std_rating="std")
            .round(3)
            .reset_index()
            .sort_values("avg_rating", ascending=False)
        )

        # By Level
        by_level = (
            courses_m.groupby("CourseLevel")["CourseRating"]
            .agg(avg_rating="mean", count="count", std_rating="std")
            .round(3)
            .reset_index()
            .sort_values("avg_rating", ascending=False)
        )

        # Category x Level heatmap pivot
        cat_level_pivot = (
            courses_m.groupby(["CourseCategory", "CourseLevel"])["CourseRating"]
            .mean()
            .round(3)
            .unstack(fill_value=np.nan)
        )

        # Gender x Level (using master which has instructor gender)
        gender_level = (
            self.master.groupby(["Gender", "CourseLevel"])["CourseRating"]
            .mean()
            .round(3)
            .unstack(fill_value=np.nan)
        )

        # Enrollment by Category x Level
        enr_pivot = (
            self.master.groupby(["CourseCategory", "CourseLevel"])["EnrollmentCount"]
            .sum()
            .unstack(fill_value=0)
        )

        return {
            "by_category": by_category,
            "by_level": by_level,
            "cat_level_pivot": cat_level_pivot,
            "gender_level": gender_level,
            "enr_pivot": enr_pivot,
        }

    # ------------------------------------------------------------------
    # Stage 5 — Instructor Impact on Course Success
    # ------------------------------------------------------------------
    def instructor_impact(
        self,
        high_thresh: float | None = None,
        low_thresh: float | None = None,
    ) -> dict[str, Any]:
        """
        Segment instructors into High/Mid/Low tiers; compare course ratings
        and enrollment volumes across tiers.
        """
        ht = high_thresh if high_thresh is not None else settings.tier_high_threshold
        lt = low_thresh if low_thresh is not None else settings.tier_low_threshold

        def tier(r: float) -> str:
            if r >= ht:
                return f"High (≥{ht:.1f})"
            elif r >= lt:
                return f"Mid ({lt:.1f}–{ht - 0.1:.1f})"
            else:
                return f"Low (<{lt:.1f})"

        m = self.master.copy()
        m["InstructorTier"] = m["TeacherRating"].apply(tier)

        tier_summary = (
            m.groupby("InstructorTier")
            .agg(
                avg_course_rating=("CourseRating", "mean"),
                avg_enrollment=("EnrollmentCount", "mean"),
                total_enrollment=("EnrollmentCount", "sum"),
                instructor_count=("TeacherID", "nunique"),
                course_count=("CourseID", "nunique"),
            )
            .round(3)
            .reset_index()
        )

        # Teacher-level leaderboard (full)
        teacher_metrics = (
            m.groupby(
                [
                    "TeacherID",
                    "TeacherName",
                    "TeacherRating",
                    "Expertise",
                    "YearsOfExperience",
                    "Gender",
                ]
            )
            .agg(
                avg_course_rating=("CourseRating", "mean"),
                total_enrollments=("EnrollmentCount", "sum"),
                course_count=("CourseID", "nunique"),
                consistency_std=("CourseRating", "std"),
            )
            .round(3)
            .reset_index()
        )
        teacher_metrics["consistency_std"] = teacher_metrics["consistency_std"].fillna(0.0)
        teacher_metrics["consistency_index"] = teacher_metrics.apply(
            lambda r: round(1.0 - r["consistency_std"] / r["avg_course_rating"], 4)
            if r["avg_course_rating"] > 0
            else 0.0,
            axis=1,
        )
        teacher_metrics["InstructorTier"] = teacher_metrics["TeacherRating"].apply(tier)
        teacher_metrics = teacher_metrics.sort_values("TeacherRating", ascending=False)

        return {
            "tier_summary": tier_summary,
            "teacher_metrics": teacher_metrics,
            "high_thresh": ht,
            "low_thresh": lt,
        }

    # ------------------------------------------------------------------
    # Stage 6 — Expertise-Based Performance Insights
    # ------------------------------------------------------------------
    def expertise_performance(self) -> dict[str, Any]:
        """
        Cross-tabulate Expertise vs course quality;
        identify domains where teaching quality is critical.
        """
        m = self.master.copy()

        # Expertise vs TeacherRating
        exp_teacher = (
            m.drop_duplicates(subset="TeacherID")
            .groupby("Expertise")["TeacherRating"]
            .agg(
                avg_teacher_rating="mean",
                std_teacher_rating="std",
                count="count",
            )
            .round(3)
            .reset_index()
            .sort_values("avg_teacher_rating", ascending=False)
        )

        # Expertise vs CourseRating (weighted by enrollment)
        exp_course = (
            m.groupby("Expertise")
            .agg(
                avg_course_rating=("CourseRating", "mean"),
                std_course_rating=("CourseRating", "std"),
                total_enrollments=("EnrollmentCount", "sum"),
                course_count=("CourseID", "nunique"),
            )
            .round(3)
            .reset_index()
            .sort_values("avg_course_rating", ascending=False)
        )

        # Correlation: TeacherRating <-> CourseRating per expertise
        exp_corr: list[dict[str, Any]] = []
        for exp, grp in m.groupby("Expertise"):
            grp_clean = grp.dropna(subset=["TeacherRating", "CourseRating"])
            if len(grp_clean) >= settings.min_group_size:
                if grp_clean["TeacherRating"].nunique() <= 1 or grp_clean["CourseRating"].nunique() <= 1:
                    r, p = np.nan, np.nan
                else:
                    r_val, p_val = stats.pearsonr(grp_clean["TeacherRating"], grp_clean["CourseRating"])
                    r, p = round(float(r_val), 4), round(float(p_val), 6)
                exp_corr.append({
                    "Expertise": exp,
                    "r": r,
                    "p": p,
                    "n": len(grp_clean),
                })
        exp_corr_df = pd.DataFrame(exp_corr).sort_values("r", ascending=False)

        # Expertise x CourseCategory heatmap
        exp_cat = (
            m.groupby(["Expertise", "CourseCategory"])["CourseRating"]
            .mean()
            .round(3)
            .unstack(fill_value=np.nan)
        )

        return {
            "exp_teacher": exp_teacher,
            "exp_course": exp_course,
            "exp_corr": exp_corr_df,
            "exp_cat_pivot": exp_cat,
        }

    # ------------------------------------------------------------------
    # Teacher <-> Course Rating correlation (core question 3)
    # ------------------------------------------------------------------
    def teacher_course_rating_correlation(self) -> dict[str, Any]:
        """
        Pearson and Spearman correlation between TeacherRating and CourseRating.
        """
        df = self.master.dropna(subset=["TeacherRating", "CourseRating"])
        r_p, p_p = stats.pearsonr(df["TeacherRating"], df["CourseRating"])
        r_s, p_s = stats.spearmanr(df["TeacherRating"], df["CourseRating"])

        # OLS slope
        X = df[["TeacherRating"]].values
        y = df["CourseRating"].values
        reg = LinearRegression().fit(X, y)
        slope = float(reg.coef_[0])
        intercept = float(reg.intercept_)

        return {
            "pearson_r": round(float(r_p), 4),
            "pearson_p": round(float(p_p), 6),
            "spearman_r": round(float(r_s), 4),
            "spearman_p": round(float(p_s), 6),
            "slope": round(slope, 6),
            "intercept": round(intercept, 6),
            "n": len(df),
        }

    # ------------------------------------------------------------------
    # Full analytics bundle
    # ------------------------------------------------------------------
    def run_full_analysis(self) -> dict[str, Any]:
        """Run all six stages + KPIs and return a single results dict."""
        return {
            "kpis": self.kpi.compute_all_kpis(),
            "per_instructor_consistency": self.kpi.per_instructor_consistency(),
            "integration_summary": self.data_integration_summary(),
            "instructor_profiles": self.instructor_profiles(),
            "experience_vs_performance": self.experience_vs_performance(),
            "course_quality": self.course_quality_evaluation(),
            "instructor_impact": self.instructor_impact(),
            "expertise_performance": self.expertise_performance(),
            "teacher_course_corr": self.teacher_course_rating_correlation(),
        }
