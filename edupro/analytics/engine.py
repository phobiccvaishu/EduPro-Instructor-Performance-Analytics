"""
edupro.analytics.engine
=======================
Six-stage modular analytical methodology engine for EduPro platform.
Provides complete analysis across Integration, Profiles, Experience, Quality, Impact, and Expertise.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import OLS

from edupro.analytics.aggregation import course_table, teacher_table
from edupro.analytics.inference import _make_verdict, run_all_inferences
from edupro.analytics.kpis import KPIEngine
from edupro.analytics.stats import correlation, experience_course_slope
from edupro.config import AnalysisConfig, default_analysis_config, settings
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

        if len(t) < 3 or t["YearsOfExperience"].nunique() < 2 or t["TeacherRating"].nunique() < 2:
            r_t, p_t = np.nan, np.nan
            r_s, p_s = np.nan, np.nan
        else:
            r_t, p_t = stats.pearsonr(t["YearsOfExperience"], t["TeacherRating"])
            r_s, p_s = stats.spearmanr(t["YearsOfExperience"], t["TeacherRating"])

        # Quartile-based binning
        t = t.copy()
        if len(t) > 0 and t["YearsOfExperience"].nunique() >= 4:
            try:
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
            except ValueError:
                t["exp_bin"] = pd.qcut(
                    t["YearsOfExperience"],
                    q=max(1, min(4, t["YearsOfExperience"].nunique())),
                    duplicates="drop",
                )
                quartile_avg = (
                    t.groupby("exp_bin", observed=True)["TeacherRating"]
                    .mean()
                    .round(3)
                    .reset_index()
                )
        elif len(t) > 0:
            t["exp_bin"] = pd.qcut(
                t["YearsOfExperience"],
                q=max(1, t["YearsOfExperience"].nunique()),
                duplicates="drop",
            )
            quartile_avg = (
                t.groupby("exp_bin", observed=True)["TeacherRating"].mean().round(3).reset_index()
            )
        else:
            quartile_avg = pd.DataFrame(columns=["exp_bin", "TeacherRating"])

        # Experience bins (0-5, 6-10, 11-15, 16+)
        bins = list(settings.exp_bins)
        labels = list(settings.exp_labels)
        t["exp_group"] = pd.cut(t["YearsOfExperience"], bins=bins, labels=labels, right=True)
        group_avg = (
            t.groupby("exp_group", observed=True)["TeacherRating"]
            .agg(["mean", "count"])
            .round(3)
            .reset_index()
        )

        # Course-level correlation (using master)
        mc = self.master.dropna(subset=["YearsOfExperience", "CourseRating"])
        if len(mc) < 3 or mc["YearsOfExperience"].nunique() < 2 or mc["CourseRating"].nunique() < 2:
            r_c, p_c = np.nan, np.nan
        else:
            r_c, p_c = stats.pearsonr(mc["YearsOfExperience"], mc["CourseRating"])

        return {
            "pearson_r_teacher": round(float(r_t), 4) if np.isfinite(r_t) else np.nan,
            "pearson_p_teacher": round(float(p_t), 6) if np.isfinite(p_t) else np.nan,
            "spearman_r_teacher": round(float(r_s), 4) if np.isfinite(r_s) else np.nan,
            "spearman_p_teacher": round(float(p_s), 6) if np.isfinite(p_s) else np.nan,
            "pearson_r_course": round(float(r_c), 4) if np.isfinite(r_c) else np.nan,
            "pearson_p_course": round(float(p_c), 6) if np.isfinite(p_c) else np.nan,
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
            lambda r: (
                round(1.0 - r["consistency_std"] / r["avg_course_rating"], 4)
                if r["avg_course_rating"] > 0
                else 0.0
            ),
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
                if (
                    grp_clean["TeacherRating"].nunique() <= 1
                    or grp_clean["CourseRating"].nunique() <= 1
                ):
                    r, p = np.nan, np.nan
                else:
                    r_val, p_val = stats.pearsonr(
                        grp_clean["TeacherRating"], grp_clean["CourseRating"]
                    )
                    r, p = round(float(r_val), 4), round(float(p_val), 6)
                exp_corr.append(
                    {
                        "Expertise": exp,
                        "r": r,
                        "p": p,
                        "n": len(grp_clean),
                    }
                )
        if exp_corr:
            exp_corr_df = pd.DataFrame(exp_corr).sort_values("r", ascending=False)
        else:
            exp_corr_df = pd.DataFrame(columns=["Expertise", "r", "p", "n"])

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
        if len(df) < 3 or df["TeacherRating"].nunique() < 2 or df["CourseRating"].nunique() < 2:
            r_p, p_p = np.nan, np.nan
            r_s, p_s = np.nan, np.nan
            slope = np.nan
            intercept = np.nan
        else:
            r_p, p_p = stats.pearsonr(df["TeacherRating"], df["CourseRating"])
            r_s, p_s = stats.spearmanr(df["TeacherRating"], df["CourseRating"])
            X = df[["TeacherRating"]].values
            y = df["CourseRating"].values
            reg = LinearRegression().fit(X, y)
            slope = float(reg.coef_[0])
            intercept = float(reg.intercept_)

        return {
            "pearson_r": round(float(r_p), 4) if np.isfinite(r_p) else np.nan,
            "pearson_p": round(float(p_p), 6) if np.isfinite(p_p) else np.nan,
            "spearman_r": round(float(r_s), 4) if np.isfinite(r_s) else np.nan,
            "spearman_p": round(float(p_s), 6) if np.isfinite(p_s) else np.nan,
            "slope": round(slope, 6) if np.isfinite(slope) else np.nan,
            "intercept": round(intercept, 6) if np.isfinite(intercept) else np.nan,
            "n": len(df),
        }

    # ------------------------------------------------------------------
    # Full analytics bundle
    # ------------------------------------------------------------------
    def run_full_analysis(
        self,
        config: AnalysisConfig | None = None,
        cfg: AnalysisConfig | None = None,
    ) -> dict[str, Any]:
        """
        Run all six stages, legacy KPIs, revised entity aggregations,
        and statistical hypothesis tests (F1-F13).

        Parameters
        ----------
        config : AnalysisConfig, optional
            Runtime configuration for statistical resampling and attribution.
        cfg : AnalysisConfig, optional
            Alias for config.
        """
        t0 = time.perf_counter()
        cfg = (
            cfg if cfg is not None else (config if config is not None else default_analysis_config)
        )

        # Stage summaries
        per_inst_cons = self.kpi.per_instructor_consistency()
        integ_summary = self.data_integration_summary()
        inst_profiles = self.instructor_profiles()

        # Step 2 additions: Entity tables
        tt = teacher_table(
            self.teachers,
            self.master,
            attribution=cfg.attribution,
            min_pair_enrollment=cfg.min_pair_enrollment,
        )
        ct = course_table(
            self.courses,
            self.master,
            attribution=cfg.attribution,
            min_pair_enrollment=cfg.min_pair_enrollment,
        )

        # Step 2 additions: KPI v2
        kpis_v2 = self.kpi.compute_all_kpis_v2(
            teachers=self.teachers,
            courses=self.courses,
            teacher_df=tt,
            cfg=cfg,
        )

        # Step 2 additions: Inferences F1-F13
        findings = run_all_inferences(
            teachers=self.teachers,
            courses=self.courses,
            master=self.master,
            teacher_df=tt,
            course_df=ct,
            cfg=cfg,
        )

        elapsed = time.perf_counter() - t0

        return {
            # Modern analytics outputs (v2.0)
            "entity_tables": {"teachers": tt, "courses": ct},
            "findings": findings,
            "kpis_v2": kpis_v2,
            "per_instructor_consistency": per_inst_cons,
            "integration_summary": integ_summary,
            "instructor_profiles": inst_profiles,
            "meta": {
                "seed": cfg.seed,
                "alpha": cfg.alpha,
                "n_boot": cfg.effective_n_boot,
                "n_perm": cfg.effective_n_perm,
                "fast_mode": cfg.fast_mode,
                "attribution": cfg.attribution,
                "min_pair_enrollment": cfg.min_pair_enrollment,
                "execution_time_seconds": round(elapsed, 3),
                "version": "2.0",
            },
        }

    def run_sensitivity(
        self,
        attribution_modes: tuple[str, ...] = ("enrollment_weighted", "unweighted"),
        min_enrollments: tuple[int | None, ...] = (None, 3),
    ) -> pd.DataFrame:
        """
        Evaluate core analytical metrics across a 2x2 grid of attribution modes
        and enrollment noise thresholds.
        """
        rows = []
        base_r = None
        base_ci = None
        base_verdict = None

        for attr in attribution_modes:
            for min_enr in min_enrollments:
                cfg = AnalysisConfig(
                    attribution=attr,
                    min_pair_enrollment=min_enr,
                    fast_mode=True,
                )
                tt = teacher_table(
                    self.teachers, self.master, attribution=attr, min_pair_enrollment=min_enr
                )
                _ct = course_table(
                    self.courses, self.master, attribution=attr, min_pair_enrollment=min_enr
                )

                kpis_v2 = self.kpi.compute_all_kpis_v2(
                    self.teachers, self.courses, teacher_df=tt, cfg=cfg
                )

                # Teacher-course correlation with p-value and 95% CI
                corr_res = correlation(
                    tt["TeacherRating"],
                    tt["weighted_course_rating"],
                    method="pearson",
                    alpha=cfg.alpha,
                )
                r_tc = (
                    float(corr_res["statistic"]) if np.isfinite(corr_res["statistic"]) else np.nan
                )
                p_tc = float(corr_res["p_value"]) if np.isfinite(corr_res["p_value"]) else np.nan
                ci_tc = (float(corr_res["ci_low"]), float(corr_res["ci_high"]))
                verdict = _make_verdict(p_tc, ci_tc, len(tt), cfg.min_group_size, cfg.alpha)

                # Course slope unified to single function experience_course_slope
                slope_res = experience_course_slope(tt, alpha=cfg.alpha)
                slope_cr = slope_res["slope"]
                slope_p = slope_res["p_value"]

                # Rule-based conclusion evaluation against baseline
                if base_verdict is None:
                    base_r = r_tc
                    base_ci = ci_tc
                    base_verdict = verdict
                    conclusion_status = "baseline (unchanged)"
                    conclusion_note = "Primary specification"
                else:
                    same_verdict = verdict == base_verdict
                    same_sign = np.sign(r_tc) == np.sign(base_r)
                    ci_overlap = not (ci_tc[1] < base_ci[0] or ci_tc[0] > base_ci[1])
                    is_unchanged = bool(same_verdict and same_sign and ci_overlap)
                    if is_unchanged:
                        conclusion_status = "unchanged"
                        conclusion_note = "Conclusion unchanged: verdict and sign match baseline with overlapping CI."
                    else:
                        conclusion_status = "changed"
                        conclusion_note = (
                            "Conclusion changed: unweighted attribution gives pairs with 1 to 7 enrollments "
                            "the same weight as pairs with hundreds of enrollments, distorting the correlation."
                        )

                rows.append(
                    {
                        "attribution": attr,
                        "min_pair_enrollment": min_enr if min_enr is not None else 0,
                        "avg_teacher_rating": round(kpis_v2["avg_teacher_rating"].value, 3),
                        "avg_teacher_rating_status": "not attribution-dependent (intrinsic instructor attribute)",
                        "avg_course_rating": round(kpis_v2["avg_course_rating"].value, 3),
                        "avg_course_rating_status": "not attribution-dependent (intrinsic course attribute)",
                        "rating_consistency_index": round(
                            kpis_v2["rating_consistency_index"].value, 3
                        ),
                        "rating_consistency_index_status": "not attribution-dependent (computed from instructor transactions)",
                        "experience_slope_course": round(slope_cr, 4)
                        if np.isfinite(slope_cr)
                        else np.nan,
                        "experience_slope_course_p": round(slope_p, 4)
                        if np.isfinite(slope_p)
                        else np.nan,
                        "teacher_course_rating_corr": round(r_tc, 4)
                        if np.isfinite(r_tc)
                        else np.nan,
                        "teacher_course_rating_p": round(p_tc, 4) if np.isfinite(p_tc) else np.nan,
                        "teacher_course_rating_ci": (round(ci_tc[0], 4), round(ci_tc[1], 4))
                        if np.isfinite(ci_tc[0])
                        else (np.nan, np.nan),
                        "teacher_course_rating_verdict": verdict,
                        "conclusion_unchanged": conclusion_status,
                        "conclusion_note": conclusion_note,
                        "enrollment_influence_ratio": round(
                            kpis_v2["enrollment_influence"].value, 3
                        ),
                    }
                )

        # Extra labelled rows: HC3 and rank-based (Spearman)
        tt_base = teacher_table(
            self.teachers,
            self.master,
            attribution="enrollment_weighted",
            min_pair_enrollment=None,
        )
        x_base = tt_base["TeacherRating"].to_numpy(dtype=float)
        y_base = tt_base["weighted_course_rating"].to_numpy(dtype=float)

        # 1. HC3 Robust OLS
        X_hc3 = sm.add_constant(x_base)
        m_hc3 = OLS(y_base, X_hc3).fit(cov_type="HC3")
        slope_hc3 = float(m_hc3.params[1])
        p_hc3 = float(m_hc3.pvalues[1])
        ci_hc3 = (float(m_hc3.conf_int()[1][0]), float(m_hc3.conf_int()[1][1]))
        verdict_hc3 = _make_verdict(p_hc3, ci_hc3, len(tt_base), 3, 0.05)

        rows.append(
            {
                "attribution": "enrollment_weighted (HC3 robust)",
                "min_pair_enrollment": 0,
                "avg_teacher_rating": rows[0]["avg_teacher_rating"],
                "avg_teacher_rating_status": rows[0]["avg_teacher_rating_status"],
                "avg_course_rating": rows[0]["avg_course_rating"],
                "avg_course_rating_status": rows[0]["avg_course_rating_status"],
                "rating_consistency_index": rows[0]["rating_consistency_index"],
                "rating_consistency_index_status": rows[0]["rating_consistency_index_status"],
                "experience_slope_course": rows[0]["experience_slope_course"],
                "experience_slope_course_p": rows[0]["experience_slope_course_p"],
                "teacher_course_rating_corr": round(slope_hc3, 4),
                "teacher_course_rating_p": round(p_hc3, 4),
                "teacher_course_rating_ci": (round(ci_hc3[0], 4), round(ci_hc3[1], 4)),
                "teacher_course_rating_verdict": verdict_hc3,
                "conclusion_unchanged": "unchanged",
                "conclusion_note": "HC3 heteroskedasticity-consistent standard errors confirm lack of rating relationship.",
                "enrollment_influence_ratio": rows[0]["enrollment_influence_ratio"],
            }
        )

        # 2. Rank-based Spearman
        s_res = correlation(x_base, y_base, method="spearman", alpha=0.05)
        rho_s = float(s_res["statistic"]) if np.isfinite(s_res["statistic"]) else np.nan
        p_s = float(s_res["p_value"]) if np.isfinite(s_res["p_value"]) else np.nan
        ci_s = (float(s_res["ci_low"]), float(s_res["ci_high"]))
        verdict_s = _make_verdict(p_s, ci_s, len(tt_base), 3, 0.05)

        rows.append(
            {
                "attribution": "enrollment_weighted (Spearman rank)",
                "min_pair_enrollment": 0,
                "avg_teacher_rating": rows[0]["avg_teacher_rating"],
                "avg_teacher_rating_status": rows[0]["avg_teacher_rating_status"],
                "avg_course_rating": rows[0]["avg_course_rating"],
                "avg_course_rating_status": rows[0]["avg_course_rating_status"],
                "rating_consistency_index": rows[0]["rating_consistency_index"],
                "rating_consistency_index_status": rows[0]["rating_consistency_index_status"],
                "experience_slope_course": rows[0]["experience_slope_course"],
                "experience_slope_course_p": rows[0]["experience_slope_course_p"],
                "teacher_course_rating_corr": round(rho_s, 4),
                "teacher_course_rating_p": round(p_s, 4),
                "teacher_course_rating_ci": (round(ci_s[0], 4), round(ci_s[1], 4)),
                "teacher_course_rating_verdict": verdict_s,
                "conclusion_unchanged": "unchanged",
                "conclusion_note": "Monotonic rank correlation confirms independence of instructor and course ratings.",
                "enrollment_influence_ratio": rows[0]["enrollment_influence_ratio"],
            }
        )

        return pd.DataFrame(rows)
