"""
edupro.analytics.kpis
=====================
KPI Calculation Engine for EduPro.
Computes all 5 key performance indicators at platform and instructor cohort levels.
Preserves existing formulas while modernizing pandas groupbys and adding type hints.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from edupro.config import settings
from edupro.logging_setup import get_logger

logger = get_logger("analytics.kpis")


class KPIEngine:
    """
    Compute the five platform KPIs at both global and filtered cohort levels.

    Parameters
    ----------
    master : pd.DataFrame
        The integrated master dataset produced by data pipeline or build_master.
    """

    def __init__(self, master: pd.DataFrame) -> None:
        self.master = master.copy()

    # ------------------------------------------------------------------
    # KPI 1 — Average Teacher Rating
    # ------------------------------------------------------------------
    def avg_teacher_rating(self) -> float:
        """Mean of TeacherRating across all instructors (deduplicated)."""
        teachers_unique = self.master.drop_duplicates(subset="TeacherID")
        return round(float(teachers_unique["TeacherRating"].mean()), 4)

    # ------------------------------------------------------------------
    # KPI 2 — Average Course Rating
    # ------------------------------------------------------------------
    def avg_course_rating(self) -> float:
        """Mean of CourseRating across all evaluated courses (deduplicated)."""
        courses_unique = self.master.drop_duplicates(subset="CourseID")
        return round(float(courses_unique["CourseRating"].mean()), 4)

    # ------------------------------------------------------------------
    # KPI 3 — Rating Consistency Index
    # ------------------------------------------------------------------
    def rating_consistency_index(self) -> float:
        """
        Platform-level Rating Consistency Index.

        Formula: 1 - (sigma_course_rating / mu_course_rating)
        """
        courses_unique = self.master.drop_duplicates(subset="CourseID")
        mu = float(courses_unique["CourseRating"].mean())
        sigma = float(courses_unique["CourseRating"].std())
        if mu == 0:
            return 0.0
        return round(1.0 - sigma / mu, 4)

    def per_instructor_consistency(self) -> pd.DataFrame:
        """
        Per-instructor Rating Consistency Index.

        Returns DataFrame with TeacherID, TeacherName, mean_course_rating,
        std_course_rating, consistency_index.
        """
        grp = (
            self.master.groupby(["TeacherID", "TeacherName"])
            .agg(
                mean_course_rating=("CourseRating", "mean"),
                std_course_rating=("CourseRating", "std"),
                course_count=("CourseRating", "count"),
            )
            .reset_index()
        )
        grp["std_course_rating"] = grp["std_course_rating"].fillna(0.0)
        grp["consistency_index"] = grp.apply(
            lambda r: round(1.0 - r["std_course_rating"] / r["mean_course_rating"], 4)
            if r["mean_course_rating"] > 0
            else 0.0,
            axis=1,
        )
        return grp.sort_values("consistency_index", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # KPI 4 — Experience Impact Score
    # ------------------------------------------------------------------
    def experience_impact_score(self) -> dict[str, Any]:
        """
        Pearson r and OLS regression slope between YearsOfExperience
        and both TeacherRating and CourseRating.

        Returns dict with keys:
          r_exp_teacher, p_exp_teacher, slope_exp_teacher,
          r_exp_course,  p_exp_course,  slope_exp_course,
          impact_score   (average of the two |r| values)
        """
        df = self.master.drop_duplicates(subset="TeacherID").dropna(
            subset=["YearsOfExperience", "TeacherRating"]
        )

        r_t, p_t = stats.pearsonr(df["YearsOfExperience"], df["TeacherRating"])

        # OLS slope for TeacherRating ~ YearsOfExperience
        X = df[["YearsOfExperience"]].values
        y_t = df["TeacherRating"].values
        reg_t = LinearRegression().fit(X, y_t)
        slope_t = float(reg_t.coef_[0])

        # Course-level: merge back to get course ratings per teacher
        df_c = self.master.dropna(subset=["YearsOfExperience", "CourseRating"])
        r_c, p_c = stats.pearsonr(df_c["YearsOfExperience"], df_c["CourseRating"])
        X_c = df_c[["YearsOfExperience"]].values
        y_c = df_c["CourseRating"].values
        reg_c = LinearRegression().fit(X_c, y_c)
        slope_c = float(reg_c.coef_[0])

        return {
            "r_exp_teacher": round(float(r_t), 4),
            "p_exp_teacher": round(float(p_t), 6),
            "slope_exp_teacher": round(slope_t, 6),
            "r_exp_course": round(float(r_c), 4),
            "p_exp_course": round(float(p_c), 6),
            "slope_exp_course": round(slope_c, 6),
            "impact_score": round((abs(float(r_t)) + abs(float(r_c))) / 2.0, 4),
        }

    # ------------------------------------------------------------------
    # KPI 5 — Enrollment Influence Ratio
    # ------------------------------------------------------------------
    def enrollment_influence_ratio(self, high_threshold: float | None = None) -> dict[str, Any]:
        """
        Ratio of avg enrollment volume for high-rated instructors vs platform avg.

        High-rated: TeacherRating >= high_threshold (default settings.tier_high_threshold = 4.0).

        Returns dict with keys:
          avg_enrollment_high, avg_enrollment_low, avg_enrollment_all,
          enrollment_influence_ratio, high_threshold
        """
        thresh = high_threshold if high_threshold is not None else settings.tier_high_threshold

        teacher_enr = (
            self.master.groupby(["TeacherID", "TeacherRating"])["EnrollmentCount"]
            .sum()
            .reset_index()
        )

        high = teacher_enr[teacher_enr["TeacherRating"] >= thresh]
        low = teacher_enr[teacher_enr["TeacherRating"] < thresh]

        avg_high = float(high["EnrollmentCount"].mean()) if len(high) > 0 else 0.0
        avg_low = float(low["EnrollmentCount"].mean()) if len(low) > 0 else 0.0
        avg_all = float(teacher_enr["EnrollmentCount"].mean())

        ratio = round(avg_high / avg_all, 4) if avg_all > 0 else 0.0

        return {
            "avg_enrollment_high": round(avg_high, 2),
            "avg_enrollment_low": round(avg_low, 2),
            "avg_enrollment_all": round(avg_all, 2),
            "enrollment_influence_ratio": ratio,
            "high_threshold": thresh,
        }

    # ------------------------------------------------------------------
    # All KPIs in one call
    # ------------------------------------------------------------------
    def compute_all_kpis(self) -> dict[str, Any]:
        """Compute all five KPIs and return results dictionary."""
        return {
            "avg_teacher_rating": self.avg_teacher_rating(),
            "avg_course_rating": self.avg_course_rating(),
            "rating_consistency_index": self.rating_consistency_index(),
            "experience_impact": self.experience_impact_score(),
            "enrollment_influence": self.enrollment_influence_ratio(),
        }
