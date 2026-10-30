"""
analytics.py
============
EduPro Online Platform — KPI Calculation Engine & Analytical Methodology
Computes all five KPIs and six analytical methodology stages.

Author  : Senior Data Scientist
Project : Instructor Performance and Course Quality Evaluation on EduPro
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")


# ============================================================================
# SECTION 1 — KPI CALCULATION ENGINE
# ============================================================================

class KPIEngine:
    """
    Compute the five required KPIs at both global and filtered cohort level.

    Parameters
    ----------
    master : pd.DataFrame
        The integrated master dataset produced by data_loader.build_master_dataset().
    """

    def __init__(self, master: pd.DataFrame):
        self.master = master.copy()

    # ------------------------------------------------------------------
    # KPI 1 — Average Teacher Rating
    # ------------------------------------------------------------------
    def avg_teacher_rating(self) -> float:
        """Mean of TeacherRating across all instructors (deduplicated)."""
        teachers_unique = self.master.drop_duplicates(subset="TeacherID")
        return round(teachers_unique["TeacherRating"].mean(), 4)

    # ------------------------------------------------------------------
    # KPI 2 — Average Course Rating
    # ------------------------------------------------------------------
    def avg_course_rating(self) -> float:
        """Mean of CourseRating across all evaluated courses (deduplicated)."""
        courses_unique = self.master.drop_duplicates(subset="CourseID")
        return round(courses_unique["CourseRating"].mean(), 4)

    # ------------------------------------------------------------------
    # KPI 3 — Rating Consistency Index
    # ------------------------------------------------------------------
    def rating_consistency_index(self) -> float:
        """
        Platform-level Rating Consistency Index.

        Formula: 1 - (σ_course_rating / μ_course_rating)

        A value close to 1 indicates uniform quality; close to 0 or negative
        signals high dispersion.  Per-instructor variant computed in
        compute_instructor_profiles().
        """
        courses_unique = self.master.drop_duplicates(subset="CourseID")
        mu = courses_unique["CourseRating"].mean()
        sigma = courses_unique["CourseRating"].std()
        if mu == 0:
            return 0.0
        return round(1 - sigma / mu, 4)

    def per_instructor_consistency(self) -> pd.DataFrame:
        """
        Per-instructor Rating Consistency Index.
        Returns DataFrame with TeacherID, TeacherName, mean_course_rating,
        std_course_rating, consistency_index.
        """
        grp = (
            self.master
            .groupby(["TeacherID", "TeacherName"])["CourseRating"]
            .agg(mean_course_rating="mean", std_course_rating="std", course_count="count")
            .reset_index()
        )
        grp["std_course_rating"] = grp["std_course_rating"].fillna(0)
        grp["consistency_index"] = grp.apply(
            lambda r: round(1 - r["std_course_rating"] / r["mean_course_rating"], 4)
            if r["mean_course_rating"] > 0 else 0,
            axis=1,
        )
        return grp.sort_values("consistency_index", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # KPI 4 — Experience Impact Score
    # ------------------------------------------------------------------
    def experience_impact_score(self) -> dict:
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
            "r_exp_teacher":    round(r_t, 4),
            "p_exp_teacher":    round(p_t, 6),
            "slope_exp_teacher": round(slope_t, 6),
            "r_exp_course":     round(r_c, 4),
            "p_exp_course":     round(p_c, 6),
            "slope_exp_course":  round(slope_c, 6),
            "impact_score":     round((abs(r_t) + abs(r_c)) / 2, 4),
        }

    # ------------------------------------------------------------------
    # KPI 5 — Enrollment Influence Ratio
    # ------------------------------------------------------------------
    def enrollment_influence_ratio(self, high_threshold: float = 4.0) -> dict:
        """
        Ratio of avg enrollment volume for high-rated instructors vs platform avg.

        High-rated: TeacherRating >= high_threshold (default 4.0).

        Returns dict with keys:
          avg_enrollment_high, avg_enrollment_low, avg_enrollment_all,
          enrollment_influence_ratio
        """
        teacher_enr = (
            self.master
            .groupby(["TeacherID", "TeacherRating"])["EnrollmentCount"]
            .sum()
            .reset_index()
        )

        high = teacher_enr[teacher_enr["TeacherRating"] >= high_threshold]
        low  = teacher_enr[teacher_enr["TeacherRating"] < high_threshold]

        avg_high = high["EnrollmentCount"].mean() if len(high) > 0 else 0
        avg_low  = low["EnrollmentCount"].mean()  if len(low) > 0  else 0
        avg_all  = teacher_enr["EnrollmentCount"].mean()

        ratio = round(avg_high / avg_all, 4) if avg_all > 0 else 0.0

        return {
            "avg_enrollment_high":       round(avg_high, 2),
            "avg_enrollment_low":        round(avg_low,  2),
            "avg_enrollment_all":        round(avg_all,  2),
            "enrollment_influence_ratio": ratio,
            "high_threshold":            high_threshold,
        }

    # ------------------------------------------------------------------
    # All KPIs in one call
    # ------------------------------------------------------------------
    def compute_all_kpis(self) -> dict:
        return {
            "avg_teacher_rating":       self.avg_teacher_rating(),
            "avg_course_rating":        self.avg_course_rating(),
            "rating_consistency_index": self.rating_consistency_index(),
            "experience_impact":        self.experience_impact_score(),
            "enrollment_influence":     self.enrollment_influence_ratio(),
        }


# ============================================================================
# SECTION 2 — ANALYTICAL METHODOLOGY (six stages)
# ============================================================================

class AnalyticsEngine:
    """
    Six-stage modular analytical engine.

    Parameters
    ----------
    master       : integrated master DataFrame
    teachers_df  : raw Teachers DataFrame
    courses_df   : raw Courses DataFrame
    transactions_df : raw Transactions DataFrame
    """

    def __init__(self,
                 master: pd.DataFrame,
                 teachers_df: pd.DataFrame,
                 courses_df: pd.DataFrame,
                 transactions_df: pd.DataFrame):
        self.master  = master.copy()
        self.teachers = teachers_df.copy()
        self.courses  = courses_df.copy()
        self.txns     = transactions_df.copy()
        self.kpi      = KPIEngine(master)

    # ------------------------------------------------------------------
    # Stage 1 — Data Integration
    # ------------------------------------------------------------------
    def data_integration_summary(self) -> dict:
        """
        Summarise the integration: unique IDs, join stats, mapping accuracy.
        """
        from data_loader import validate_mapping
        report = validate_mapping(self.master)
        report["teacher_ids_in_transactions"] = self.txns["TeacherID"].nunique()
        report["course_ids_in_transactions"]  = self.txns["CourseID"].nunique()
        return report

    # ------------------------------------------------------------------
    # Stage 2 — Instructor Profile Analysis
    # ------------------------------------------------------------------
    def instructor_profiles(self) -> dict:
        """
        Distributions of Age, YearsOfExperience, Expertise;
        rating spread; top/low performers.
        """
        t = self.teachers.copy()

        rating_stats = t["TeacherRating"].describe().round(3).to_dict()

        # Top 10 performers
        top10 = (
            t.sort_values("TeacherRating", ascending=False)
            .head(10)[["TeacherID", "TeacherName", "Expertise",
                        "YearsOfExperience", "TeacherRating"]]
            .reset_index(drop=True)
        )

        # Bottom 10 performers
        bottom10 = (
            t.sort_values("TeacherRating", ascending=True)
            .head(10)[["TeacherID", "TeacherName", "Expertise",
                        "YearsOfExperience", "TeacherRating"]]
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
            "rating_stats":    rating_stats,
            "top10":           top10,
            "bottom10":        bottom10,
            "expertise_dist":  expertise_dist,
        }

    # ------------------------------------------------------------------
    # Stage 3 — Experience vs. Performance Analysis
    # ------------------------------------------------------------------
    def experience_vs_performance(self) -> dict:
        """
        Correlations between YearsOfExperience and ratings;
        experience threshold / diminishing returns analysis.
        """
        t = self.teachers.dropna(subset=["YearsOfExperience", "TeacherRating"])

        r_t, p_t = stats.pearsonr(t["YearsOfExperience"], t["TeacherRating"])
        r_s, p_s = stats.spearmanr(t["YearsOfExperience"], t["TeacherRating"])

        # Quartile-based binning
        t = t.copy()
        t["exp_bin"] = pd.qcut(t["YearsOfExperience"], q=4,
                                labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"],
                                duplicates="drop")
        quartile_avg = (
            t.groupby("exp_bin", observed=True)["TeacherRating"]
            .mean()
            .round(3)
            .reset_index()
        )

        # Experience bins (0-5, 6-10, 11-15, 16+)
        bins   = [0, 5, 10, 15, 100]
        labels = ["0–5 yrs", "6–10 yrs", "11–15 yrs", "16+ yrs"]
        t["exp_group"] = pd.cut(t["YearsOfExperience"], bins=bins,
                                labels=labels, right=True)
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
            "pearson_r_teacher":   round(r_t, 4),
            "pearson_p_teacher":   round(p_t, 6),
            "spearman_r_teacher":  round(r_s, 4),
            "spearman_p_teacher":  round(p_s, 6),
            "pearson_r_course":    round(r_c, 4),
            "pearson_p_course":    round(p_c, 6),
            "quartile_avg":        quartile_avg,
            "group_avg":           group_avg,
        }

    # ------------------------------------------------------------------
    # Stage 4 — Course Quality Evaluation
    # ------------------------------------------------------------------
    def course_quality_evaluation(self) -> dict:
        """
        CourseRating by Category and Level; Gender × Level comparison.
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

        # Category × Level heatmap pivot
        cat_level_pivot = (
            courses_m.groupby(["CourseCategory", "CourseLevel"])["CourseRating"]
            .mean()
            .round(3)
            .unstack(fill_value=np.nan)
        )

        # Gender × Level (using master which has instructor gender)
        gender_level = (
            self.master.groupby(["Gender", "CourseLevel"])["CourseRating"]
            .mean()
            .round(3)
            .unstack(fill_value=np.nan)
        )

        # Enrollment by Category × Level
        enr_pivot = (
            self.master.groupby(["CourseCategory", "CourseLevel"])["EnrollmentCount"]
            .sum()
            .unstack(fill_value=0)
        )

        return {
            "by_category":      by_category,
            "by_level":         by_level,
            "cat_level_pivot":  cat_level_pivot,
            "gender_level":     gender_level,
            "enr_pivot":        enr_pivot,
        }

    # ------------------------------------------------------------------
    # Stage 5 — Instructor Impact on Course Success
    # ------------------------------------------------------------------
    def instructor_impact(self, high_thresh: float = 4.0,
                           low_thresh: float = 3.0) -> dict:
        """
        Segment instructors into High/Mid/Low tiers; compare course ratings
        and enrollment volumes across tiers.
        """
        def tier(r):
            if r >= high_thresh:
                return "High (≥4.0)"
            elif r >= low_thresh:
                return "Mid (3.0–3.9)"
            else:
                return "Low (<3.0)"

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
            m.groupby(["TeacherID", "TeacherName", "TeacherRating",
                        "Expertise", "YearsOfExperience", "Gender"])
            .agg(
                avg_course_rating=("CourseRating", "mean"),
                total_enrollments=("EnrollmentCount", "sum"),
                course_count=("CourseID", "nunique"),
                consistency_std=("CourseRating", "std"),
            )
            .round(3)
            .reset_index()
        )
        teacher_metrics["consistency_std"] = teacher_metrics["consistency_std"].fillna(0)
        teacher_metrics["consistency_index"] = teacher_metrics.apply(
            lambda r: round(1 - r["consistency_std"] / r["avg_course_rating"], 4)
            if r["avg_course_rating"] > 0 else 0, axis=1
        )
        teacher_metrics["InstructorTier"] = teacher_metrics["TeacherRating"].apply(tier)
        teacher_metrics = teacher_metrics.sort_values("TeacherRating", ascending=False)

        return {
            "tier_summary":    tier_summary,
            "teacher_metrics": teacher_metrics,
            "high_thresh":     high_thresh,
            "low_thresh":      low_thresh,
        }

    # ------------------------------------------------------------------
    # Stage 6 — Expertise-Based Performance Insights
    # ------------------------------------------------------------------
    def expertise_performance(self) -> dict:
        """
        Cross-tabulate Expertise vs course quality;
        identify domains where teaching quality is critical.
        """
        m = self.master.copy()

        # Expertise vs TeacherRating
        exp_teacher = (
            m.drop_duplicates(subset="TeacherID")
            .groupby("Expertise")["TeacherRating"]
            .agg(avg_teacher_rating="mean", std_teacher_rating="std",
                 count="count")
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

        # Correlation: TeacherRating ↔ CourseRating per expertise
        exp_corr = []
        for exp, grp in m.groupby("Expertise"):
            grp_clean = grp.dropna(subset=["TeacherRating", "CourseRating"])
            if len(grp_clean) >= 3:
                r, p = stats.pearsonr(grp_clean["TeacherRating"],
                                      grp_clean["CourseRating"])
                exp_corr.append({"Expertise": exp, "r": round(r, 4),
                                  "p": round(p, 6), "n": len(grp_clean)})
        exp_corr_df = pd.DataFrame(exp_corr).sort_values("r", ascending=False)

        # Expertise × CourseCategory heatmap
        exp_cat = (
            m.groupby(["Expertise", "CourseCategory"])["CourseRating"]
            .mean()
            .round(3)
            .unstack(fill_value=np.nan)
        )

        return {
            "exp_teacher":    exp_teacher,
            "exp_course":     exp_course,
            "exp_corr":       exp_corr_df,
            "exp_cat_pivot":  exp_cat,
        }

    # ------------------------------------------------------------------
    # Teacher ↔ Course Rating correlation (core question 3)
    # ------------------------------------------------------------------
    def teacher_course_rating_correlation(self) -> dict:
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
            "pearson_r":  round(r_p, 4),
            "pearson_p":  round(p_p, 6),
            "spearman_r": round(r_s, 4),
            "spearman_p": round(p_s, 6),
            "slope":      round(slope, 6),
            "intercept":  round(intercept, 6),
            "n":          len(df),
        }

    # ------------------------------------------------------------------
    # Full analytics bundle
    # ------------------------------------------------------------------
    def run_full_analysis(self) -> dict:
        """Run all six stages + KPIs and return a single results dict."""
        return {
            "kpis":                      self.kpi.compute_all_kpis(),
            "per_instructor_consistency": self.kpi.per_instructor_consistency(),
            "integration_summary":       self.data_integration_summary(),
            "instructor_profiles":       self.instructor_profiles(),
            "experience_vs_performance": self.experience_vs_performance(),
            "course_quality":            self.course_quality_evaluation(),
            "instructor_impact":         self.instructor_impact(),
            "expertise_performance":     self.expertise_performance(),
            "teacher_course_corr":       self.teacher_course_rating_correlation(),
        }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_loader import get_all_data
    data = get_all_data()
    engine = AnalyticsEngine(
        data["master"], data["teachers"],
        data["courses"], data["transactions"],
    )
    results = engine.run_full_analysis()
    print("KPIs:")
    for k, v in results["kpis"].items():
        print(f"  {k}: {v}")
    print("\nTeacher-Course Rating Correlation:")
    for k, v in results["teacher_course_corr"].items():
        print(f"  {k}: {v}")
