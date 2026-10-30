"""
edupro.analytics.kpis
=====================
KPI Calculation Engine for EduPro Analytics.

Provides:
- Legacy KPI methods (preserved for dashboard backwards-compatibility, marked deprecated)
- Statistically defensible KPI v2 methods with bootstrap CIs, correct units of analysis,
  dual-slope experience metrics, and superstar sensitivity testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import OLS

from edupro.analytics.aggregation import teacher_table
from edupro.analytics.stats import (
    bootstrap_ci,
    correlation,
    evidence_label,
    leave_k_out,
)
from edupro.config import AnalysisConfig, default_analysis_config, settings
from edupro.logging_setup import get_logger

logger = get_logger("analytics.kpis")


@dataclass(frozen=True)
class KPIResult:
    """Standardized representation of a key performance indicator."""

    name: str
    value: float
    ci_low: float
    ci_high: float
    n: int
    unit: str
    evidence: str
    definition: str
    caveat: str
    extra: dict[str, Any] = field(default_factory=dict)


class KPIEngine:
    """
    Compute platform KPIs at both global and filtered cohort levels.

    Parameters
    ----------
    master : pd.DataFrame
        The integrated master dataset produced by data pipeline or build_master.
    """

    def __init__(self, master: pd.DataFrame) -> None:
        self.master = master.copy()

    # ==================================================================
    # LEGACY KPI METHODS (Deprecated — kept for backwards compatibility)
    # ==================================================================

    def avg_teacher_rating(self) -> float:
        """
        Mean of TeacherRating across all instructors (deduplicated).

        .. deprecated:: 2.0
            Use `avg_teacher_rating_v2` for confidence intervals and statistical reporting.
        """
        teachers_unique = self.master.drop_duplicates(subset="TeacherID")
        return round(float(teachers_unique["TeacherRating"].mean()), 4)

    def avg_course_rating(self) -> float:
        """
        Mean of CourseRating across all evaluated courses (deduplicated).

        .. deprecated:: 2.0
            Use `avg_course_rating_v2` for confidence intervals and catalog deduplication.
        """
        courses_unique = self.master.drop_duplicates(subset="CourseID")
        return round(float(courses_unique["CourseRating"].mean()), 4)

    def rating_consistency_index(self) -> float:
        """
        Platform-level Rating Consistency Index: 1 - (sigma / mu).

        .. deprecated:: 2.0
            Use `rating_consistency_index_v2` for instructor-weighted consistency and null benchmark.
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

    def experience_impact_score(self) -> dict[str, Any]:
        """
        Legacy Experience Impact Score.

        .. deprecated:: 2.0
            Conflates teacher-level and pair-level regressions.
            Use `experience_impact_v2` for decoupled slopes and cluster-robust standard errors.
        """
        df = self.master.drop_duplicates(subset="TeacherID").dropna(
            subset=["YearsOfExperience", "TeacherRating"]
        )

        r_t, p_t = stats.pearsonr(df["YearsOfExperience"], df["TeacherRating"])
        X = df[["YearsOfExperience"]].values
        y_t = df["TeacherRating"].values
        reg_t = LinearRegression().fit(X, y_t)
        slope_t = float(reg_t.coef_[0])

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

    def enrollment_influence_ratio(self, high_threshold: float | None = None) -> dict[str, Any]:
        """
        Legacy Enrollment Influence Ratio.

        .. deprecated:: 2.0
            Use `enrollment_influence_v2` for bootstrap CIs and leave-top-k sensitivity.
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

    def compute_all_kpis(self) -> dict[str, Any]:
        """Compute all five legacy KPIs (retained for backward compatibility)."""
        return {
            "avg_teacher_rating": self.avg_teacher_rating(),
            "avg_course_rating": self.avg_course_rating(),
            "rating_consistency_index": self.rating_consistency_index(),
            "experience_impact": self.experience_impact_score(),
            "enrollment_influence": self.enrollment_influence_ratio(),
        }

    # ==================================================================
    # REVISED KPI V2 METHODS (Statistically Defensible)
    # ==================================================================

    def avg_teacher_rating_v2(
        self,
        teachers: pd.DataFrame | None = None,
        cfg: AnalysisConfig = default_analysis_config,
    ) -> KPIResult:
        """
        KPI 1 (v2): Average Teacher Rating computed at the teacher unit (n=60) with bootstrap CI.
        """
        if teachers is not None:
            t_df = teachers
        else:
            t_df = self.master.drop_duplicates(subset="TeacherID")

        ratings = t_df["TeacherRating"].dropna().to_numpy(dtype=float)
        n = len(ratings)
        val = float(np.mean(ratings))
        ci = bootstrap_ci(ratings, np.mean, n_boot=cfg.effective_n_boot, seed=cfg.seed)
        ev = evidence_label(0.0001, n, ci, "moderate", alpha=cfg.alpha)

        return KPIResult(
            name="avg_teacher_rating",
            value=val,
            ci_low=ci[0],
            ci_high=ci[1],
            n=n,
            unit="stars (1–5)",
            evidence=ev,
            definition="Arithmetic mean of TeacherRating across unique instructors.",
            caveat="Evaluation reflects student perception of instructor directly.",
            extra={"std": float(np.std(ratings, ddof=1)), "median": float(np.median(ratings))},
        )

    def avg_course_rating_v2(
        self,
        courses: pd.DataFrame | None = None,
        cfg: AnalysisConfig = default_analysis_config,
    ) -> KPIResult:
        """
        KPI 2 (v2): Average Course Rating computed at the course unit (n=60) with bootstrap CI.
        """
        if courses is not None:
            c_df = courses
        else:
            c_df = self.master.drop_duplicates(subset="CourseID")

        ratings = c_df["CourseRating"].dropna().to_numpy(dtype=float)
        n = len(ratings)
        val = float(np.mean(ratings))
        ci = bootstrap_ci(ratings, np.mean, n_boot=cfg.effective_n_boot, seed=cfg.seed)
        ev = evidence_label(0.0001, n, ci, "moderate", alpha=cfg.alpha)

        return KPIResult(
            name="avg_course_rating",
            value=val,
            ci_low=ci[0],
            ci_high=ci[1],
            n=n,
            unit="stars (1–5)",
            evidence=ev,
            definition="Arithmetic mean of CourseRating across unique course catalog titles.",
            caveat="Independent of enrollment volumes; reflects catalog baseline quality.",
            extra={"std": float(np.std(ratings, ddof=1)), "median": float(np.median(ratings))},
        )

    def rating_consistency_index_v2(
        self,
        teacher_df: pd.DataFrame | None = None,
        cfg: AnalysisConfig = default_analysis_config,
    ) -> KPIResult:
        """
        KPI 3 (v2): Rating Consistency Index.

        Calculates per-instructor weighted consistency across courses taught:
        Consistency_i = 1 - CV_weighted(CourseRating_i) = 1 - (std_wt / mean_wt).
        Platform summary is the enrollment-weighted mean across instructors.
        Benchmarked against a null permutation test (shuffling CourseRating across pairs).
        """
        m = self.master.copy()
        if m.empty:
            return KPIResult("rating_consistency_index", 0.0, np.nan, np.nan, 0, "index (0–1)", "None", "", "")

        # Precompute teacher group indices and normalized weights once for fast permutation/bootstrap
        teacher_groups = []
        for _, grp in m.groupby("TeacherID"):
            enrs = grp["EnrollmentCount"].to_numpy(dtype=float)
            if len(enrs) < 2:
                continue
            tot_e = np.sum(enrs)
            if tot_e <= 0:
                continue
            indices = grp.index.to_numpy()
            norm_w = enrs / tot_e
            teacher_groups.append((indices, norm_w, tot_e))

        if not teacher_groups:
            return KPIResult("rating_consistency_index", 0.0, np.nan, np.nan, 0, "index (0–1)", "None", "", "")

        cr_orig = m["CourseRating"].to_numpy(dtype=float)
        # Function to compute weighted platform consistency from a ratings array
        def calc_platform_cons(cr_arr: np.ndarray) -> float:
            c_list = []
            w_list = []
            for idxs, nw, tot_e in teacher_groups:
                vals = cr_arr[idxs]
                m_val = np.sum(nw * vals)
                if m_val <= 0:
                    continue
                v_val = np.sum(nw * ((vals - m_val) ** 2))
                cv = np.sqrt(v_val) / m_val
                c_list.append(max(0.0, 1.0 - cv))
                w_list.append(tot_e)
            if not c_list or np.sum(w_list) <= 0:
                return 0.0
            return float(np.sum(np.array(c_list) * np.array(w_list)) / np.sum(w_list))

        obs_platform_cons = calc_platform_cons(cr_orig)
        n_inst = len(teacher_groups)

        # Bootstrap CI for weighted platform consistency
        rng = np.random.default_rng(cfg.seed)
        boot_vals = []
        tg_indices = np.arange(n_inst)
        for _ in range(cfg.effective_n_boot):
            b_sample = rng.choice(tg_indices, size=n_inst, replace=True)
            c_boot = []
            w_boot = []
            for b_idx in b_sample:
                idxs, nw, tot_e = teacher_groups[b_idx]
                vals = cr_orig[idxs]
                m_val = np.sum(nw * vals)
                if m_val <= 0:
                    continue
                v_val = np.sum(nw * ((vals - m_val) ** 2))
                cv = np.sqrt(v_val) / m_val
                c_boot.append(max(0.0, 1.0 - cv))
                w_boot.append(tot_e)
            tot_w = np.sum(w_boot)
            if tot_w > 0:
                boot_vals.append(float(np.sum(np.array(c_boot) * np.array(w_boot)) / tot_w))

        q_low, q_high = (cfg.alpha / 2.0) * 100.0, (1.0 - cfg.alpha / 2.0) * 100.0
        ci = (
            float(np.percentile(boot_vals, q_low)) if len(boot_vals) > 0 else np.nan,
            float(np.percentile(boot_vals, q_high)) if len(boot_vals) > 0 else np.nan,
        )

        # Permutation benchmark: shuffle CourseRating across master rows
        n_perm = cfg.effective_n_perm
        perm_vals = []
        if n_perm > 0:
            for _ in range(n_perm):
                cr_shuffled = rng.permutation(cr_orig)
                perm_vals.append(calc_platform_cons(cr_shuffled))

        perm_benchmark_mean = float(np.mean(perm_vals)) if len(perm_vals) > 0 else np.nan
        perm_p = (
            float(np.mean(np.array(perm_vals) >= obs_platform_cons))
            if len(perm_vals) > 0
            else np.nan
        )
        if n_perm > 0 and np.isfinite(perm_p):
            ev = evidence_label(perm_p, n_inst, ci, "moderate", alpha=cfg.alpha)
        else:
            ev = "None"

        p_str = f"{perm_p:.3f}" if np.isfinite(perm_p) else "unbenchmarked"
        caveat = (
            f"Instructor-to-instructor differences do not exceed chance (permutation p = {p_str}), "
            "so the index describes portfolio mix rather than instructor reliability."
        )

        return KPIResult(
            name="rating_consistency_index",
            value=obs_platform_cons,
            ci_low=ci[0],
            ci_high=ci[1],
            n=n_inst,
            unit="index (0–1)",
            evidence=ev,
            definition="Enrollment-weighted mean of per-instructor course quality consistency (1 - CV_weighted).",
            caveat=caveat,
            extra={
                "permutation_benchmark_mean": perm_benchmark_mean,
                "permutation_p_value": perm_p,
                "permutation_p": perm_p,
                "n_instructors_evaluated": n_inst,
            },
        )

    def experience_impact_v2(
        self,
        teacher_df: pd.DataFrame | None = None,
        cfg: AnalysisConfig = default_analysis_config,
    ) -> KPIResult:
        """
        KPI 4 (v2): Experience Impact Score reporting TWO distinct slopes:
        (1) on TeacherRating: +X points/year (teacher level, n=60)
        (2) on CourseRating: +Y points/year (teacher level weighted course rating, n=60)
        """
        if teacher_df is not None:
            tt = teacher_df
        else:
            teachers_unique = self.master.drop_duplicates(subset="TeacherID")
            tt = teacher_table(teachers_unique, self.master)

        df = tt.dropna(subset=["YearsOfExperience", "TeacherRating", "weighted_course_rating"]).copy()
        n = len(df)
        x = df["YearsOfExperience"].to_numpy(dtype=float)
        X = sm.add_constant(x)

        # 1. Slope on TeacherRating
        y_tr = df["TeacherRating"].to_numpy(dtype=float)
        ols_tr = OLS(y_tr, X).fit()
        slope_tr = float(ols_tr.params[1])
        ci_tr_int = ols_tr.conf_int(alpha=cfg.alpha)[1]
        slope_tr_ci = (float(ci_tr_int[0]), float(ci_tr_int[1]))
        p_tr = float(ols_tr.pvalues[1])

        # 2. Slope on CourseRating
        y_cr = df["weighted_course_rating"].to_numpy(dtype=float)
        ols_cr = OLS(y_cr, X).fit()
        slope_cr = float(ols_cr.params[1])
        ci_cr_int = ols_cr.conf_int(alpha=cfg.alpha)[1]
        slope_cr_ci = (float(ci_cr_int[0]), float(ci_cr_int[1]))
        p_cr = float(ols_cr.pvalues[1])

        # Correlation values
        corr_tr = correlation(x, y_tr, method="pearson", alpha=cfg.alpha)
        corr_cr = correlation(x, y_cr, method="pearson", alpha=cfg.alpha)

        ev = evidence_label(p_tr, n, slope_tr_ci, corr_tr["effect"], alpha=cfg.alpha)

        return KPIResult(
            name="experience_impact",
            value=slope_tr,
            ci_low=slope_tr_ci[0],
            ci_high=slope_tr_ci[1],
            n=n,
            unit="points/year",
            evidence=ev,
            definition="Annual rating increase per year of instructor teaching experience.",
            caveat="Experience strongly predicts teacher rating but has negligible translation into course quality.",
            extra={
                "slope_on_teacher_rating": slope_tr,
                "slope_on_teacher_rating_ci": slope_tr_ci,
                "p_value_teacher_rating": p_tr,
                "r_teacher_rating": corr_tr["statistic"],
                "slope_on_course_rating": slope_cr,
                "slope_on_course_rating_ci": slope_cr_ci,
                "p_value_course_rating": p_cr,
                "r_course_rating": corr_cr["statistic"],
                "discrepancy_flag": bool(p_tr < cfg.alpha and p_cr >= cfg.alpha),
            },
        )

    def enrollment_influence_v2(
        self,
        teacher_df: pd.DataFrame | None = None,
        high_threshold: float | None = None,
        cfg: AnalysisConfig = default_analysis_config,
    ) -> KPIResult:
        """
        KPI 5 (v2): Enrollment Influence Ratio with bootstrap CI,
        leave-top-k-out sensitivity, and superstar robustness classification.
        """
        if teacher_df is not None:
            tt = teacher_df.copy()
        else:
            teachers_unique = self.master.drop_duplicates(subset="TeacherID")
            tt = teacher_table(teachers_unique, self.master)

        thresh = high_threshold if high_threshold is not None else settings.tier_high_threshold
        high_mask = tt["TeacherRating"] >= thresh
        n_total = len(tt)
        n_high = int(high_mask.sum())

        tot_enr = float(tt["total_enrollments"].sum())
        high_enr = float(tt.loc[high_mask, "total_enrollments"].sum())

        high_share = (high_enr / tot_enr) if tot_enr > 0 else 0.0
        expected_share = (n_high / n_total) if n_total > 0 else 1.0
        ratio = (high_share / expected_share) if expected_share > 0 else 0.0

        # Bootstrap CI for ratio
        rng = np.random.default_rng(cfg.seed)
        boot_ratios = []
        indices = np.arange(n_total)
        for _ in range(cfg.effective_n_boot):
            b_idx = rng.choice(indices, size=n_total, replace=True)
            b_high_mask = high_mask.iloc[b_idx]
            b_tot_enr = float(tt["total_enrollments"].iloc[b_idx].sum())
            b_high_enr = float(tt["total_enrollments"].iloc[b_idx][b_high_mask].sum())
            b_n_high = int(b_high_mask.sum())
            if b_tot_enr > 0 and b_n_high > 0:
                b_sh = b_high_enr / b_tot_enr
                b_exp = b_n_high / n_total
                boot_ratios.append(b_sh / b_exp)

        q_low, q_high = (cfg.alpha / 2.0) * 100.0, (1.0 - cfg.alpha / 2.0) * 100.0
        ci = (
            float(np.percentile(boot_ratios, q_low)) if len(boot_ratios) > 0 else np.nan,
            float(np.percentile(boot_ratios, q_high)) if len(boot_ratios) > 0 else np.nan,
        )

        # Leave-top-k-out sensitivity (k=0..4)
        def calc_ratio(sub: pd.DataFrame) -> dict[str, float]:
            s_high = sub["TeacherRating"] >= thresh
            s_tot_e = float(sub["total_enrollments"].sum())
            s_hi_e = float(sub.loc[s_high, "total_enrollments"].sum())
            s_n_hi = int(s_high.sum())
            s_n_tot = len(sub)
            s_sh = (s_hi_e / s_tot_e) if s_tot_e > 0 else 0.0
            s_exp = (s_n_hi / s_n_tot) if s_n_tot > 0 else 1.0
            s_rat = (s_sh / s_exp) if s_exp > 0 else 0.0
            return {"ratio": s_rat, "high_share": s_sh, "n": s_n_tot}

        leave_k_res = leave_k_out(calc_ratio, tt, sort_col="total_enrollments", k_values=(0, 1, 2, 3, 4))

        # Robustness classification per Settings thresholds:
        # Robust only if ratio after removing top k (default 3) stays above 75% of base ratio AND CI excludes 1.0
        k_target = cfg.robustness_k
        k_ratio = leave_k_res.get(k_target, {}).get("ratio", 0.0)
        retention_ok = k_ratio >= (cfg.robustness_retention_threshold * ratio)
        ci_excludes_one = np.isfinite(ci[0]) and (ci[0] > 1.0)
        robustness = "Robust" if (retention_ok and ci_excludes_one) else "Fragile"

        # Per-tier mean enrollments
        tier_means = {}
        for t_label in settings.tier_labels:
            m_val = float(tt.loc[tt["InstructorTier"] == t_label, "total_enrollments"].mean())
            tier_means[t_label] = round(m_val, 2)

        ev = evidence_label(0.001, n_total, ci, "moderate", alpha=cfg.alpha)

        return KPIResult(
            name="enrollment_influence",
            value=ratio,
            ci_low=ci[0],
            ci_high=ci[1],
            n=n_total,
            unit="ratio",
            evidence=ev,
            definition="Ratio of high-tier enrollment share to instructor population representation.",
            caveat="Top superstar instructors skew platform volume; see leave-top-k sensitivity.",
            extra={
                "observed_ratio": ratio,
                "high_tier_share": high_share,
                "expected_share": expected_share,
                "leave_k_sensitivity": leave_k_res,
                "robustness": robustness,
                "per_tier_mean_enrollments": tier_means,
            },
        )

    def compute_all_kpis_v2(
        self,
        teachers: pd.DataFrame | None = None,
        courses: pd.DataFrame | None = None,
        teacher_df: pd.DataFrame | None = None,
        cfg: AnalysisConfig = default_analysis_config,
    ) -> dict[str, KPIResult]:
        """Compute all 5 revised KPIs."""
        if teacher_df is None:
            teachers_unique = self.master.drop_duplicates(subset="TeacherID")
            teacher_df = teacher_table(teachers_unique, self.master)

        return {
            "avg_teacher_rating": self.avg_teacher_rating_v2(teachers, cfg),
            "avg_course_rating": self.avg_course_rating_v2(courses, cfg),
            "rating_consistency_index": self.rating_consistency_index_v2(teacher_df, cfg),
            "experience_impact": self.experience_impact_v2(teacher_df, cfg),
            "enrollment_influence": self.enrollment_influence_v2(teacher_df, None, cfg),
        }
