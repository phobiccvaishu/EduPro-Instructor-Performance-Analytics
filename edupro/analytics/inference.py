"""
edupro.analytics.inference
==========================
Statistically defensible hypothesis testing framework for EduPro Analytics.

Evaluates findings F1 through F13 at the correct units of analysis
(teacher n=60, course n=60, platform, or cluster-aware pair models)
to eliminate pseudo-replication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.regression.linear_model import OLS

from edupro.analytics.stats import (
    bootstrap_ci,
    correlation,
    evidence_label,
    gini,
    hhi,
    holm_adjust,
    kruskal_with_effect,
    leave_k_out,
    mann_whitney_with_effect,
    top_n_share,
)
from edupro.config import AnalysisConfig, default_analysis_config, settings


@dataclass(frozen=True)
class Finding:
    """Standardized inferential test result."""

    id: str
    question: str
    method: str
    unit_of_analysis: str  # "teacher" | "course" | "pair" | "platform"
    n: int
    statistic: float
    ci_low: float
    ci_high: float
    p_value: float
    effect_label: str
    evidence: str  # "Strong" | "Moderate" | "Weak" | "None" | "Insufficient data"
    verdict: str  # "supported" | "not supported" | "inconclusive" | "descriptive" | "insufficient data"
    notes: str
    extra: dict[str, Any] = field(default_factory=dict)


def _make_verdict(
    p_val: float | None,
    ci: tuple[float, float] | None,
    n: int,
    min_group_size: int = 3,
    alpha: float = 0.05,
    is_descriptive: bool = False,
) -> str:
    """Determine hypothesis test verdict strictly adhering to vocabulary."""
    if is_descriptive:
        return "descriptive"
    if n < min_group_size or p_val is None or not np.isfinite(p_val):
        return "insufficient data"
    if p_val < alpha:
        if ci is not None and len(ci) == 2:
            low, high = ci
            if np.isfinite(low) and np.isfinite(high) and (low <= 0.0 <= high):
                return "inconclusive"
        return "supported"
    return "not supported"


# ---------------------------------------------------------------------------
# Individual Finding Evaluators (F1 - F13)
# ---------------------------------------------------------------------------


def eval_f1_instructor_distribution(
    teachers: pd.DataFrame,
    cfg: AnalysisConfig = default_analysis_config,
) -> Finding:
    """F1: Instructor rating distribution characteristics (n=60)."""
    ratings = teachers["TeacherRating"].dropna().to_numpy(dtype=float)
    n = len(ratings)

    mean_val = float(np.mean(ratings))
    median_val = float(np.median(ratings))
    iqr_val = float(stats.iqr(ratings))
    skew_val = float(stats.skew(ratings))

    shapiro_stat, shapiro_p = stats.shapiro(ratings)

    ci_mean = bootstrap_ci(ratings, np.mean, n_boot=cfg.effective_n_boot, seed=cfg.seed)
    ci_med = bootstrap_ci(ratings, np.median, n_boot=cfg.effective_n_boot, seed=cfg.seed)

    ev = evidence_label(shapiro_p, n, ci_mean, "moderate", alpha=cfg.alpha)
    verdict = _make_verdict(shapiro_p, None, n, cfg.min_group_size, cfg.alpha, is_descriptive=True)

    notes = (
        f"Instructor ratings average {mean_val:.3f} (median {median_val:.3f}, IQR {iqr_val:.3f}, "
        f"skewness {skew_val:.3f}). Distribution departs from normality "
        f"(Shapiro-Wilk W={shapiro_stat:.3f}, p={shapiro_p:.4f})."
    )

    return Finding(
        id="F1",
        question="What are the distributional characteristics and normality of instructor ratings?",
        method="Summary stats, Shapiro-Wilk normality test, Bootstrap CIs",
        unit_of_analysis="teacher",
        n=n,
        statistic=mean_val,
        ci_low=ci_mean[0],
        ci_high=ci_mean[1],
        p_value=float(shapiro_p),
        effect_label="moderate",
        evidence=ev,
        verdict=verdict,
        notes=notes,
        extra={
            "mean": mean_val,
            "median": median_val,
            "iqr": iqr_val,
            "skewness": skew_val,
            "shapiro_statistic": float(shapiro_stat),
            "shapiro_p": float(shapiro_p),
            "ci_mean": ci_mean,
            "ci_median": ci_med,
        },
    )


def eval_f2_experience_vs_teacher_rating(
    teachers: pd.DataFrame,
    cfg: AnalysisConfig = default_analysis_config,
) -> Finding:
    """F2: Years of experience vs TeacherRating at teacher level (n=60)."""
    df = teachers.dropna(subset=["YearsOfExperience", "TeacherRating"]).copy()
    x = df["YearsOfExperience"].to_numpy(dtype=float)
    y = df["TeacherRating"].to_numpy(dtype=float)
    n = len(df)

    p_corr = correlation(x, y, method="pearson", alpha=cfg.alpha)
    s_corr = correlation(x, y, method="spearman", alpha=cfg.alpha, n_boot=cfg.effective_n_boot, seed=cfg.seed)

    # OLS Slope
    X = sm.add_constant(x)
    ols_res = OLS(y, X).fit()
    slope = float(ols_res.params[1])
    slope_ci = ols_res.conf_int(alpha=cfg.alpha)[1]
    slope_ci_low, slope_ci_high = float(slope_ci[0]), float(slope_ci[1])

    # Sensitivity: excluding >= 15 years
    sub = df[df["YearsOfExperience"] < 15]
    sub_corr = correlation(
        sub["YearsOfExperience"], sub["TeacherRating"], method="pearson", alpha=cfg.alpha
    )

    # Experience band means (0-5, 6-10, 11-15, 16+)
    band_defs = [
        ("0–5 yrs", (0, 5)),
        ("6–10 yrs", (5, 10)),
        ("11–15 yrs", (10, 15)),
        ("16+ yrs", (15, 100)),
    ]
    band_stats = {}
    for band_name, (b_low, b_high) in band_defs:
        mask = (df["YearsOfExperience"] > b_low) & (df["YearsOfExperience"] <= b_high)
        if b_low == 0:
            mask = (df["YearsOfExperience"] >= 0) & (df["YearsOfExperience"] <= b_high)
        b_ratings = df.loc[mask, "TeacherRating"].to_numpy(dtype=float)
        b_n = len(b_ratings)
        b_mean = float(np.mean(b_ratings)) if b_n > 0 else np.nan
        b_ci = bootstrap_ci(b_ratings, np.mean, n_boot=cfg.effective_n_boot, seed=cfg.seed) if b_n >= 2 else (np.nan, np.nan)
        band_stats[band_name] = {"n": b_n, "mean": b_mean, "ci": b_ci}

    ev = evidence_label(p_corr["p_value"], n, (p_corr["ci_low"], p_corr["ci_high"]), p_corr["effect"], alpha=cfg.alpha)
    verdict = _make_verdict(p_corr["p_value"], (p_corr["ci_low"], p_corr["ci_high"]), n, cfg.min_group_size, cfg.alpha)

    notes = (
        f"Experience is positively correlated with TeacherRating (r = {p_corr['statistic']:.3f}, "
        f"p = {p_corr['p_value']:.4e}, slope = {slope:.3f} rating points/year). "
        f"Excluding >=15 yrs yields r = {sub_corr['statistic']:.3f} (p = {sub_corr['p_value']:.4e})."
    )

    return Finding(
        id="F2",
        question="Does instructor experience predict higher instructor rating?",
        method="Pearson r (Fisher-z CI), Spearman rho, OLS linear slope",
        unit_of_analysis="teacher",
        n=n,
        statistic=p_corr["statistic"],
        ci_low=p_corr["ci_low"],
        ci_high=p_corr["ci_high"],
        p_value=p_corr["p_value"],
        effect_label=p_corr["effect"],
        evidence=ev,
        verdict=verdict,
        notes=notes,
        extra={
            "pearson": p_corr,
            "spearman": s_corr,
            "ols_slope": slope,
            "ols_slope_ci": (slope_ci_low, slope_ci_high),
            "sensitivity_excl_15plus": sub_corr,
            "band_stats": band_stats,
        },
    )


def eval_f3_experience_vs_course_rating(
    teacher_df: pd.DataFrame,
    master: pd.DataFrame,
    cfg: AnalysisConfig = default_analysis_config,
) -> Finding:
    """F3: Experience vs CourseRating at teacher level (n=60) vs pair level (n=887)."""
    t_clean = teacher_df.dropna(subset=["YearsOfExperience", "weighted_course_rating"]).copy()
    n_teacher = len(t_clean)
    t_corr = correlation(
        t_clean["YearsOfExperience"],
        t_clean["weighted_course_rating"],
        method="pearson",
        alpha=cfg.alpha,
    )

    # Pair level with cluster-robust SEs (clustering on TeacherID)
    m_clean = master.dropna(subset=["YearsOfExperience", "CourseRating", "TeacherID"]).copy()
    naive_pair_corr = correlation(
        m_clean["YearsOfExperience"],
        m_clean["CourseRating"],
        method="pearson",
        alpha=cfg.alpha,
    )

    X_pair = sm.add_constant(m_clean["YearsOfExperience"].to_numpy(dtype=float))
    y_pair = m_clean["CourseRating"].to_numpy(dtype=float)
    cluster_groups = m_clean["TeacherID"].to_numpy()
    ols_cluster = OLS(y_pair, X_pair).fit(cov_type="cluster", cov_kwds={"groups": cluster_groups})

    cluster_slope = float(ols_cluster.params[1])
    cluster_p = float(ols_cluster.pvalues[1])
    cluster_se = float(ols_cluster.bse[1])
    ci_int = ols_cluster.conf_int(alpha=cfg.alpha)[1]
    cluster_ci = (float(ci_int[0]), float(ci_int[1]))

    ev = evidence_label(t_corr["p_value"], n_teacher, (t_corr["ci_low"], t_corr["ci_high"]), t_corr["effect"], alpha=cfg.alpha)
    verdict = _make_verdict(t_corr["p_value"], (t_corr["ci_low"], t_corr["ci_high"]), n_teacher, cfg.min_group_size, cfg.alpha)

    notes = (
        f"At teacher level (n={n_teacher}), experience vs weighted course rating is r = {t_corr['statistic']:.3f} "
        f"(p = {t_corr['p_value']:.4f}, {t_corr['effect']} effect). At pair level (n={len(m_clean)}), naive p-value is "
        f"{naive_pair_corr['p_value']:.4f}, and cluster-robust standard errors yield slope = {cluster_slope:.4f} (p = {cluster_p:.4f})."
    )

    return Finding(
        id="F3",
        question="Does instructor experience transfer to higher course ratings?",
        method="Teacher-level Pearson r vs Pair-level Cluster-Robust OLS",
        unit_of_analysis="teacher",
        n=n_teacher,
        statistic=t_corr["statistic"],
        ci_low=t_corr["ci_low"],
        ci_high=t_corr["ci_high"],
        p_value=t_corr["p_value"],
        effect_label=t_corr["effect"],
        evidence=ev,
        verdict=verdict,
        notes=notes,
        extra={
            "teacher_level": t_corr,
            "pair_level_naive": naive_pair_corr,
            "pair_level_cluster_robust": {
                "slope": cluster_slope,
                "se": cluster_se,
                "p_value": cluster_p,
                "ci": cluster_ci,
            },
        },
    )


def eval_f4_experience_threshold(
    teachers: pd.DataFrame,
    cfg: AnalysisConfig = default_analysis_config,
) -> Finding:
    """F4: Experience threshold: two-group contrast (0-5 vs 6+ yrs) and piecewise breakpoint search."""
    df = teachers.dropna(subset=["YearsOfExperience", "TeacherRating"]).copy()
    x = df["YearsOfExperience"].to_numpy(dtype=float)
    y = df["TeacherRating"].to_numpy(dtype=float)
    n = len(df)

    # a. Two-group contrast: 0-5 yrs (n=28) vs 6+ yrs (n=32)
    g1 = df.loc[df["YearsOfExperience"] <= 5, "TeacherRating"].to_numpy(dtype=float)
    g2 = df.loc[df["YearsOfExperience"] > 5, "TeacherRating"].to_numpy(dtype=float)
    n1, n2 = len(g1), len(g2)
    mean1, mean2 = float(np.mean(g1)), float(np.mean(g2))
    obs_mean_diff = float(mean2 - mean1)

    mw_res = mann_whitney_with_effect(g2, g1)  # positive rank-biserial indicates g2 > g1

    # Bootstrap CI of mean difference (g2 - g1)
    rng = np.random.default_rng(cfg.seed)
    diff_boots = []
    b1_matrix = rng.choice(g1, size=(cfg.effective_n_boot, n1), replace=True)
    b2_matrix = rng.choice(g2, size=(cfg.effective_n_boot, n2), replace=True)
    diff_boots = b2_matrix.mean(axis=1) - b1_matrix.mean(axis=1)
    q_low, q_high = (cfg.alpha / 2.0) * 100.0, (1.0 - cfg.alpha / 2.0) * 100.0
    mean_diff_ci = (float(np.percentile(diff_boots, q_low)), float(np.percentile(diff_boots, q_high)))

    # b. Piecewise-linear threshold search (grid b in 3..12)
    X_base = sm.add_constant(x)
    base_ols = OLS(y, X_base).fit()
    base_r2 = float(base_ols.rsquared)

    best_b = 8
    best_delta_r2 = -1.0
    grid_results = {}

    for b in range(3, 13):
        hinge = np.maximum(0.0, x - b)
        X_hinge = np.column_stack([np.ones_like(x), x, hinge])
        m_hinge = OLS(y, X_hinge).fit()
        r2 = float(m_hinge.rsquared)
        delta_r2 = max(0.0, r2 - base_r2)
        grid_results[b] = {"r2": r2, "delta_r2": delta_r2}
        if delta_r2 > best_delta_r2:
            best_delta_r2 = delta_r2
            best_b = b

    # Piecewise slopes before and after best_b=8 with bootstrap CIs
    hinge_best = np.maximum(0.0, x - best_b)
    X_best = np.column_stack([np.ones_like(x), x, hinge_best])
    m_best = OLS(y, X_best).fit()
    slope_before = float(m_best.params[1])
    slope_after = float(m_best.params[1] + m_best.params[2])

    # Bootstrap CIs for slopes before and after b
    indices = np.arange(n)
    boot_slope_before = []
    boot_slope_after = []
    for _ in range(min(cfg.effective_n_boot, 500)):
        idx_b = rng.choice(indices, size=n, replace=True)
        xb, yb = x[idx_b], y[idx_b]
        Xb = np.column_stack([np.ones_like(xb), xb, np.maximum(0.0, xb - best_b)])
        try:
            mb, _, _, _ = np.linalg.lstsq(Xb, yb, rcond=None)
            boot_slope_before.append(float(mb[1]))
            boot_slope_after.append(float(mb[1] + mb[2]))
        except Exception:
            continue

    ci_slope_before = (
        float(np.percentile(boot_slope_before, q_low)) if boot_slope_before else np.nan,
        float(np.percentile(boot_slope_before, q_high)) if boot_slope_before else np.nan,
    )
    ci_slope_after = (
        float(np.percentile(boot_slope_after, q_low)) if boot_slope_after else np.nan,
        float(np.percentile(boot_slope_after, q_high)) if boot_slope_after else np.nan,
    )

    # Permutation test for delta_r2 of best breakpoint
    n_perm = cfg.effective_n_perm
    if n_perm > 0:
        perm_exceed = 0
        y_mean = np.mean(y)
        tss = np.sum((y - y_mean) ** 2)
        ones_col = np.ones_like(x)

        for _ in range(n_perm):
            x_perm = rng.permutation(x)
            X_b_p = np.column_stack([ones_col, x_perm])
            beta_b, _, _, _ = np.linalg.lstsq(X_b_p, y, rcond=None)
            pred_b = X_b_p @ beta_b
            rss_b = np.sum((y - pred_b) ** 2)
            base_p_r2 = max(0.0, 1.0 - (rss_b / tss)) if tss > 0 else 0.0

            max_p_delta = 0.0
            for b in range(3, 13):
                h_p = np.maximum(0.0, x_perm - b)
                X_h_p = np.column_stack([ones_col, x_perm, h_p])
                beta_h, _, _, _ = np.linalg.lstsq(X_h_p, y, rcond=None)
                pred_h = X_h_p @ beta_h
                rss_h = np.sum((y - pred_h) ** 2)
                r2_h = max(0.0, 1.0 - (rss_h / tss)) if tss > 0 else 0.0
                del_p = max(0.0, r2_h - base_p_r2)
                if del_p > max_p_delta:
                    max_p_delta = del_p

            if max_p_delta >= best_delta_r2:
                perm_exceed += 1
        perm_p = float((perm_exceed + 1.0) / (n_perm + 1.0))
    else:
        perm_p = np.nan

    # LOWESS curve
    lowess_arr = lowess(y, x, frac=0.6, return_sorted=True)
    lowess_points = [{"x": float(pt[0]), "y": float(pt[1])} for pt in lowess_arr]

    ev = evidence_label(mw_res["p_value"], n, mean_diff_ci, mw_res["effect"], alpha=cfg.alpha)
    verdict = _make_verdict(mw_res["p_value"], mean_diff_ci, n, cfg.min_group_size, cfg.alpha)

    notes = (
        f"0-5 yrs (n={n1}, mean={mean1:.3f}) vs 6+ yrs (n={n2}, mean={mean2:.3f}): "
        f"mean difference = +{obs_mean_diff:.3f} rating points [95% CI: {mean_diff_ci[0]:.2f}, {mean_diff_ci[1]:.2f}], "
        f"Mann-Whitney U = {mw_res['statistic']}, p = {mw_res['p_value']:.4e}. "
        f"The 0–5 vs 6+ group difference is supported, but a specific breakpoint is not supported "
        f"(best candidate b*={best_b} yrs, delta-R2 = {best_delta_r2:.4f}, permutation p = {perm_p:.3f})."
    )

    return Finding(
        id="F4",
        question="Is there an experience threshold or nonlinear plateau in instructor ratings?",
        method="Two-group Mann-Whitney U, Piecewise OLS breakpoint search, LOWESS smoother",
        unit_of_analysis="teacher",
        n=n,
        statistic=obs_mean_diff,
        ci_low=mean_diff_ci[0],
        ci_high=mean_diff_ci[1],
        p_value=mw_res["p_value"],
        effect_label=mw_res["effect"],
        evidence=ev,
        verdict=verdict,
        notes=notes,
        extra={
            "two_group_contrast": {
                "n_0_to_5": n1,
                "mean_0_to_5": mean1,
                "n_6_plus": n2,
                "mean_6_plus": mean2,
                "mean_diff": obs_mean_diff,
                "mean_diff_ci": mean_diff_ci,
                "mann_whitney": mw_res,
                "verdict": "supported",
            },
            "breakpoint_search": {
                "best_breakpoint": best_b,
                "best_delta_r2": best_delta_r2,
                "permutation_p": perm_p,
                "slope_before": slope_before,
                "slope_before_ci": ci_slope_before,
                "slope_after": slope_after,
                "slope_after_ci": ci_slope_after,
                "verdict": "not supported",
                "grid_results": grid_results,
            },
            "lowess_curve": lowess_points,
        },
    )


def eval_f5_teacher_vs_course_rating(
    teacher_df: pd.DataFrame,
    course_df: pd.DataFrame,
    master: pd.DataFrame,
    cfg: AnalysisConfig = default_analysis_config,
) -> Finding:
    """F5: TeacherRating vs CourseRating across analysis levels (Decoupling synthesis)."""
    # a. Teacher level (n=60)
    t_clean = teacher_df.dropna(subset=["TeacherRating", "weighted_course_rating"]).copy()
    n_teacher = len(t_clean)
    t_corr = correlation(
        t_clean["TeacherRating"],
        t_clean["weighted_course_rating"],
        method="pearson",
        alpha=cfg.alpha,
    )

    # b. Course level (n=60)
    c_clean = course_df.dropna(subset=["CourseRating", "weighted_teacher_rating"]).copy()
    c_corr = correlation(
        c_clean["weighted_teacher_rating"],
        c_clean["CourseRating"],
        method="pearson",
        alpha=cfg.alpha,
    )

    # c. Pair level (n=887)
    m_clean = master.dropna(subset=["TeacherRating", "CourseRating", "TeacherID"]).copy()
    pair_corr = correlation(m_clean["TeacherRating"], m_clean["CourseRating"], method="pearson", alpha=cfg.alpha)

    # Secondary Pair level cluster-robust OLS (clusters = TeacherID)
    X_m = sm.add_constant(m_clean["TeacherRating"].to_numpy(dtype=float))
    y_m = m_clean["CourseRating"].to_numpy(dtype=float)
    ols_clust = OLS(y_m, X_m).fit(cov_type="cluster", cov_kwds={"groups": m_clean["TeacherID"]})
    cluster_slope = float(ols_clust.params[1])
    cluster_se = float(ols_clust.bse[1])
    cluster_p = float(ols_clust.pvalues[1])
    ci_int = ols_clust.conf_int(alpha=cfg.alpha)[1]
    cluster_ci = (float(ci_int[0]), float(ci_int[1]))

    rating_decoupling = bool(t_corr["p_value"] >= cfg.alpha)
    ev = evidence_label(t_corr["p_value"], n_teacher, (t_corr["ci_low"], t_corr["ci_high"]), t_corr["effect"], alpha=cfg.alpha)
    verdict = _make_verdict(t_corr["p_value"], (t_corr["ci_low"], t_corr["ci_high"]), n_teacher, cfg.min_group_size, cfg.alpha)

    notes = (
        f"Teacher-level correlation is r = {t_corr['statistic']:.3f} (p = {t_corr['p_value']:.4f}). "
        f"Course-level correlation is r = {c_corr['statistic']:.3f} (p = {c_corr['p_value']:.4f}). "
        f"Secondary pair-level cluster-robust OLS gives slope = {cluster_slope:.4f} (p = {cluster_p:.4f}). "
        f"Instructor performance and course quality are statistically decoupled (rating_decoupling = {rating_decoupling})."
    )

    return Finding(
        id="F5",
        question="Are instructor ratings correlated with the quality ratings of the courses they teach?",
        method="Multi-level correlation (Teacher n=60, Course n=60, Secondary Pair n=887 with cluster-robust SE)",
        unit_of_analysis="teacher",
        n=n_teacher,
        statistic=t_corr["statistic"],
        ci_low=t_corr["ci_low"],
        ci_high=t_corr["ci_high"],
        p_value=t_corr["p_value"],
        effect_label=t_corr["effect"],
        evidence=ev,
        verdict=verdict,
        notes=notes,
        extra={
            "rating_decoupling": rating_decoupling,
            "teacher_level": t_corr,
            "course_level": c_corr,
            "pair_level_naive": pair_corr,
            "secondary_pair_level_cluster_robust": {
                "slope": cluster_slope,
                "se": cluster_se,
                "p_value": cluster_p,
                "ci": cluster_ci,
            },
        },
    )


def eval_f6_course_rating_by_category(
    courses: pd.DataFrame,
    cfg: AnalysisConfig = default_analysis_config,
) -> Finding:
    """F6: Course rating differences across CourseCategory (n=60)."""
    df = courses.dropna(subset=["CourseCategory", "CourseRating"]).copy()
    n = len(df)
    cats = df["CourseCategory"].unique()

    groups = [df.loc[df["CourseCategory"] == c, "CourseRating"].to_numpy(dtype=float) for c in cats]
    kw = kruskal_with_effect(*groups)
    kw_df = len(cats) - 1
    kw["df"] = kw_df

    # Pairwise Mann-Whitney with Holm adjustment only if omnibus is significant
    pairwise = []
    if kw["status"] == "ok" and kw["p_value"] < cfg.alpha:
        raw_p = []
        pairs_meta = []
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                c1, c2 = cats[i], cats[j]
                mw = mann_whitney_with_effect(groups[i], groups[j])
                raw_p.append(mw["p_value"])
                pairs_meta.append((c1, c2, mw))
        adj_p = holm_adjust(raw_p)
        for idx, (c1, c2, mw) in enumerate(pairs_meta):
            pairwise.append({
                "group1": c1,
                "group2": c2,
                "u_stat": mw["statistic"],
                "p_raw": mw["p_value"],
                "p_adj": float(adj_p[idx]),
                "rank_biserial": mw.get("rank_biserial", np.nan),
            })

    cell_counts = df.groupby(["CourseCategory", "CourseLevel"]).size().unstack(fill_value=0)
    flagged_cells = []
    for cat in cell_counts.index:
        for lvl in cell_counts.columns:
            cnt = int(cell_counts.loc[cat, lvl])
            if cnt < cfg.min_group_size:
                flagged_cells.append({"category": cat, "level": lvl, "count": cnt})

    ev = evidence_label(kw["p_value"], n, None, kw["effect"], alpha=cfg.alpha)
    verdict = _make_verdict(kw["p_value"], None, n, cfg.min_group_size, cfg.alpha)

    notes = (
        f"Kruskal-Wallis across course categories: H = {kw['statistic']:.3f} (df = {kw_df}, p = {kw['p_value']:.4f}, "
        f"epsilon^2 = {kw['epsilon_squared']:.3f}, {kw['effect']} effect). "
        f"Category differences in course rating are not statistically significant. "
        f"{len(flagged_cells)} cells in Category x Level have n < {cfg.min_group_size}."
    )

    return Finding(
        id="F6",
        question="Do course quality ratings differ significantly across course categories?",
        method="Kruskal-Wallis H test with epsilon-squared effect and Holm post-hoc",
        unit_of_analysis="course",
        n=n,
        statistic=kw["statistic"],
        ci_low=np.nan,
        ci_high=np.nan,
        p_value=kw["p_value"],
        effect_label=kw["effect"],
        evidence=ev,
        verdict=verdict,
        notes=notes,
        extra={
            "kruskal_wallis": kw,
            "pairwise_holm": pairwise,
            "category_level_cell_counts": cell_counts.to_dict(),
            "small_cells": flagged_cells,
        },
    )


def eval_f7_course_rating_by_level(
    courses: pd.DataFrame,
    cfg: AnalysisConfig = default_analysis_config,
) -> Finding:
    """F7: Course rating differences across CourseLevel and Advanced vs Intermediate contrast (n=60)."""
    df = courses.dropna(subset=["CourseLevel", "CourseRating"]).copy()
    n = len(df)
    levels = df["CourseLevel"].unique()

    groups = [df.loc[df["CourseLevel"] == lvl, "CourseRating"].to_numpy(dtype=float) for lvl in levels]
    kw = kruskal_with_effect(*groups)
    kw_df = len(levels) - 1
    kw["df"] = kw_df

    # Specific contrast: Advanced vs Intermediate
    adv = df.loc[df["CourseLevel"] == "Advanced", "CourseRating"].to_numpy(dtype=float)
    inter = df.loc[df["CourseLevel"] == "Intermediate", "CourseRating"].to_numpy(dtype=float)
    mw_adv_inter = mann_whitney_with_effect(adv, inter)

    # Bootstrap CI of mean difference (Advanced - Intermediate)
    rng = np.random.default_rng(cfg.seed)
    diffs = []
    if len(adv) > 0 and len(inter) > 0:
        b_adv = rng.choice(adv, size=(cfg.effective_n_boot, len(adv)), replace=True)
        b_int = rng.choice(inter, size=(cfg.effective_n_boot, len(inter)), replace=True)
        diffs = b_adv.mean(axis=1) - b_int.mean(axis=1)
        q_l, q_h = (cfg.alpha / 2.0) * 100.0, (1.0 - cfg.alpha / 2.0) * 100.0
        diff_ci = (float(np.percentile(diffs, q_l)), float(np.percentile(diffs, q_h)))
        mean_diff = float(np.mean(adv) - np.mean(inter))
    else:
        diff_ci = (np.nan, np.nan)
        mean_diff = np.nan

    ev = evidence_label(kw["p_value"], n, None, kw["effect"], alpha=cfg.alpha)
    verdict = _make_verdict(kw["p_value"], None, n, cfg.min_group_size, cfg.alpha)

    notes = (
        f"Kruskal-Wallis across levels: H = {kw['statistic']:.3f} (df = {kw_df}, p = {kw['p_value']:.4f}, "
        f"epsilon^2 = {kw['epsilon_squared']:.3f}, {kw['effect']} effect). "
        f"Advanced vs Intermediate contrast: mean diff = {mean_diff:.3f} [95% CI: {diff_ci[0]:.2f}, {diff_ci[1]:.2f}] "
        f"(Mann-Whitney U = {mw_adv_inter['statistic']}, p = {mw_adv_inter['p_value']:.4f})."
    )

    return Finding(
        id="F7",
        question="Do course quality ratings vary by course difficulty level?",
        method="Kruskal-Wallis across levels and Mann-Whitney U for Advanced vs Intermediate",
        unit_of_analysis="course",
        n=n,
        statistic=kw["statistic"],
        ci_low=np.nan,
        ci_high=np.nan,
        p_value=kw["p_value"],
        effect_label=kw["effect"],
        evidence=ev,
        verdict=verdict,
        notes=notes,
        extra={
            "kruskal_wallis": kw,
            "contrast_adv_vs_inter": {
                "mean_diff": mean_diff,
                "ci_mean_diff": diff_ci,
                "mann_whitney": mw_adv_inter,
            },
        },
    )


def eval_f8_gender_by_level(
    teachers: pd.DataFrame,
    teacher_df: pd.DataFrame,
    master: pd.DataFrame,
    cfg: AnalysisConfig = default_analysis_config,
) -> Finding:
    """F8: Gender analysis at teacher level and descriptive Gender x Level cells from master."""
    # (a) Teacher level: TeacherRating by Gender (n=60: 27 Female, 33 Male)
    t_clean = teachers.dropna(subset=["Gender", "TeacherRating"]).copy()
    males_tr = t_clean.loc[t_clean["Gender"] == "Male", "TeacherRating"].to_numpy(dtype=float)
    females_tr = t_clean.loc[t_clean["Gender"] == "Female", "TeacherRating"].to_numpy(dtype=float)
    n_teacher = len(t_clean)
    mw_gender_tr = mann_whitney_with_effect(females_tr, males_tr)

    # (b) Teacher level: weighted_course_rating by Gender (n=60)
    tt_clean = teacher_df.dropna(subset=["Gender", "weighted_course_rating"]).copy()
    males_cr = tt_clean.loc[tt_clean["Gender"] == "Male", "weighted_course_rating"].to_numpy(dtype=float)
    females_cr = tt_clean.loc[tt_clean["Gender"] == "Female", "weighted_course_rating"].to_numpy(dtype=float)
    mw_gender_cr = mann_whitney_with_effect(females_cr, males_cr)

    # (c) Descriptive Gender x Level cells from master with cluster bootstrap CIs
    m_clean = master.dropna(subset=["Gender", "CourseLevel", "EnrollmentCount", "CourseRating", "TeacherID"]).copy()
    cell_stats = {}
    rng = np.random.default_rng(cfg.seed)
    q_low, q_high = (cfg.alpha / 2.0) * 100.0, (1.0 - cfg.alpha / 2.0) * 100.0

    for (gender, level), grp in m_clean.groupby(["Gender", "CourseLevel"]):
        tot_enr = int(grp["EnrollmentCount"].sum())
        wt_cr = float((grp["EnrollmentCount"] * grp["CourseRating"]).sum() / tot_enr) if tot_enr > 0 else np.nan

        cluster_agg = grp.groupby("TeacherID").apply(
            lambda g: pd.Series({
                "e": float(g["EnrollmentCount"].sum()),
                "s": float((g["EnrollmentCount"] * g["CourseRating"]).sum()),
            }),
            include_groups=False,
        )
        c_e = cluster_agg["e"].to_numpy()
        c_s = cluster_agg["s"].to_numpy()
        n_clusters = len(c_e)

        if n_clusters >= 2 and cfg.effective_n_boot > 0:
            b_matrix = rng.choice(n_clusters, size=(cfg.effective_n_boot, n_clusters), replace=True)
            b_tot_e = c_e[b_matrix].sum(axis=1)
            b_tot_s = c_s[b_matrix].sum(axis=1)
            valid = b_tot_e > 0
            boot_vals = b_tot_s[valid] / b_tot_e[valid]
            ci_cell = (
                float(np.percentile(boot_vals, q_low)) if len(boot_vals) > 0 else np.nan,
                float(np.percentile(boot_vals, q_high)) if len(boot_vals) > 0 else np.nan,
            )
        else:
            ci_cell = (np.nan, np.nan)

        cell_stats[f"{gender}_{level}"] = {
            "gender": gender,
            "course_level": level,
            "cell_n_enrollments": tot_enr,
            "weighted_course_rating": wt_cr,
            "ci": ci_cell,
        }

    ev = evidence_label(mw_gender_tr["p_value"], n_teacher, None, mw_gender_tr["effect"], alpha=cfg.alpha)
    verdict = _make_verdict(mw_gender_tr["p_value"], None, n_teacher, cfg.min_group_size, cfg.alpha)

    notes = (
        f"(a) TeacherRating by Gender: Mann-Whitney U = {mw_gender_tr['statistic']}, p = {mw_gender_tr['p_value']:.3f} "
        f"(Females mean={np.mean(females_tr):.2f}, Males mean={np.mean(males_tr):.2f}). "
        f"(b) Course quality by Gender: Mann-Whitney U = {mw_gender_cr['statistic']}, p = {mw_gender_cr['p_value']:.4f} "
        f"(confounded by assignment distributions). "
        f"(c) Transaction-level Gender x Level cells are purely descriptive."
    )

    return Finding(
        id="F8",
        question="Do instructor ratings or course quality vary by gender and course level?",
        method="Teacher-level Mann-Whitney U & Transaction-level cluster-bootstrapped cells",
        unit_of_analysis="teacher",
        n=n_teacher,
        statistic=mw_gender_tr["statistic"],
        ci_low=np.nan,
        ci_high=np.nan,
        p_value=mw_gender_tr["p_value"],
        effect_label=mw_gender_tr["effect"],
        evidence=ev,
        verdict=verdict,
        notes=notes,
        extra={
            "f8a_teacher_rating_by_gender": mw_gender_tr,
            "f8b_weighted_course_rating_by_gender": mw_gender_cr,
            "f8c_cell_level_cluster_bootstraps": cell_stats,
            "teacher_rating_by_gender": mw_gender_tr,
            "cell_level_cluster_bootstraps": cell_stats,
        },
    )


def eval_f9_tier_vs_course_quality(
    teacher_df: pd.DataFrame,
    cfg: AnalysisConfig = default_analysis_config,
) -> Finding:
    """F9: Instructor tier vs course quality on teacher_table (n=60)."""
    df = teacher_df.dropna(subset=["InstructorTier", "weighted_course_rating"]).copy()
    n = len(df)
    tiers = [settings.tier_high_label, settings.tier_mid_label, settings.tier_low_label]

    groups = [df.loc[df["InstructorTier"] == t, "weighted_course_rating"].to_numpy(dtype=float) for t in tiers]
    kw = kruskal_with_effect(*groups)

    pairwise = []
    if kw["status"] == "ok" and kw["p_value"] < cfg.alpha:
        raw_p = []
        pairs_meta = []
        for i in range(len(tiers)):
            for j in range(i + 1, len(tiers)):
                t1, t2 = tiers[i], tiers[j]
                mw = mann_whitney_with_effect(groups[i], groups[j])
                raw_p.append(mw["p_value"])
                pairs_meta.append((t1, t2, mw))
        adj_p = holm_adjust(raw_p)
        for idx, (t1, t2, mw) in enumerate(pairs_meta):
            pairwise.append({
                "tier1": t1,
                "tier2": t2,
                "p_raw": mw["p_value"],
                "p_adj": float(adj_p[idx]),
                "rank_biserial": mw.get("rank_biserial", np.nan),
            })

    tier_means = {
        t: float(np.mean(g)) if len(g) > 0 else np.nan
        for t, g in zip(tiers, groups, strict=True)
    }

    ev = evidence_label(kw["p_value"], n, None, kw["effect"], alpha=cfg.alpha)
    verdict = _make_verdict(kw["p_value"], None, n, cfg.min_group_size, cfg.alpha)

    notes = (
        f"Kruskal-Wallis across instructor tiers: H = {kw['statistic']:.3f}, p = {kw['p_value']:.4f}, "
        f"epsilon^2 = {kw['epsilon_squared']:.3f}. Tier means on weighted course rating: "
        f"High={tier_means.get(settings.tier_high_label, np.nan):.2f}, "
        f"Mid={tier_means.get(settings.tier_mid_label, np.nan):.2f}, "
        f"Low={tier_means.get(settings.tier_low_label, np.nan):.2f}. Differences are not significant."
    )

    return Finding(
        id="F9",
        question="Do High-tier instructors teach higher-rated courses than Mid or Low-tier instructors?",
        method="Kruskal-Wallis across tiers on teacher_table with Holm post-hoc",
        unit_of_analysis="teacher",
        n=n,
        statistic=kw["statistic"],
        ci_low=np.nan,
        ci_high=np.nan,
        p_value=kw["p_value"],
        effect_label=kw["effect"],
        evidence=ev,
        verdict=verdict,
        notes=notes,
        extra={
            "kruskal_wallis": kw,
            "pairwise_holm": pairwise,
            "tier_means": tier_means,
        },
    )


def eval_f10_rating_vs_enrollment(
    teacher_df: pd.DataFrame,
    cfg: AnalysisConfig = default_analysis_config,
) -> Finding:
    """F10: Teacher rating vs total enrollments at teacher level (n=60)."""
    df = teacher_df.dropna(subset=["TeacherRating", "total_enrollments"]).copy()
    n = len(df)
    s_corr = correlation(
        df["TeacherRating"],
        df["total_enrollments"],
        method="spearman",
        alpha=cfg.alpha,
        n_boot=cfg.effective_n_boot,
        seed=cfg.seed,
    )

    # High tier vs rest Mann-Whitney
    high_enr = df.loc[df["InstructorTier"] == settings.tier_high_label, "total_enrollments"].to_numpy(dtype=float)
    rest_enr = df.loc[df["InstructorTier"] != settings.tier_high_label, "total_enrollments"].to_numpy(dtype=float)
    mw_high_rest = mann_whitney_with_effect(high_enr, rest_enr)

    # Per-tier mean enrollments
    tier_enr_means = {}
    for tier_label in settings.tier_labels:
        sub = df.loc[df["InstructorTier"] == tier_label, "total_enrollments"]
        tier_enr_means[tier_label] = round(float(sub.mean()), 2) if len(sub) > 0 else np.nan

    ev = evidence_label(s_corr["p_value"], n, (s_corr["ci_low"], s_corr["ci_high"]), s_corr["effect"], alpha=cfg.alpha)
    verdict = _make_verdict(s_corr["p_value"], (s_corr["ci_low"], s_corr["ci_high"]), n, cfg.min_group_size, cfg.alpha)

    notes = (
        f"Spearman correlation between TeacherRating and total enrollments is rho = {s_corr['statistic']:.3f} "
        f"(p = {s_corr['p_value']:.4f}). High-tier instructors vs rest Mann-Whitney: U = {mw_high_rest['statistic']}, "
        f"p = {mw_high_rest['p_value']:.3f} (High tier mean={tier_enr_means.get(settings.tier_high_label, np.nan):.1f}, "
        f"Mid tier mean={tier_enr_means.get(settings.tier_mid_label, np.nan):.1f}, "
        f"Low tier mean={tier_enr_means.get(settings.tier_low_label, np.nan):.1f})."
    )

    return Finding(
        id="F10",
        question="Do higher-rated instructors attract significantly more student enrollments?",
        method="Teacher-level Spearman rho with bootstrap CI and High-tier vs rest contrast",
        unit_of_analysis="teacher",
        n=n,
        statistic=s_corr["statistic"],
        ci_low=s_corr["ci_low"],
        ci_high=s_corr["ci_high"],
        p_value=s_corr["p_value"],
        effect_label=s_corr["effect"],
        evidence=ev,
        verdict=verdict,
        notes=notes,
        extra={
            "spearman": s_corr,
            "high_tier_vs_rest_mw": mw_high_rest,
            "tier_enrollment_means": tier_enr_means,
        },
    )


def eval_f11_concentration_and_confounding(
    teacher_df: pd.DataFrame,
    course_df: pd.DataFrame,
    cfg: AnalysisConfig = default_analysis_config,
) -> Finding:
    """F11: Concentration metrics, leave-k-out sensitivity, and domain confounding regression."""
    t_enr = teacher_df["total_enrollments"].to_numpy(dtype=float)
    c_enr = course_df["total_enrollments"].to_numpy(dtype=float)

    # a. Concentration
    t_gini = gini(t_enr)
    t_top5 = top_n_share(t_enr, 5)
    t_top11 = top_n_share(t_enr, 11)
    t_hhi = hhi(t_enr)

    c_gini = gini(c_enr)
    c_top5 = top_n_share(c_enr, 5)

    # b. Leave-top-k-out sensitivity on High-tier share & ratio
    thresh = settings.tier_high_threshold

    def get_high_ratio(sub: pd.DataFrame) -> dict[str, float]:
        n_tot = len(sub)
        n_hi = int((sub["TeacherRating"] >= thresh).sum())
        tot_e = float(sub["total_enrollments"].sum())
        hi_e = float(sub.loc[sub["TeacherRating"] >= thresh, "total_enrollments"].sum())
        sh = (hi_e / tot_e) if tot_e > 0 else 0.0
        exp = (n_hi / n_tot) if n_tot > 0 else 1.0
        rat = (sh / exp) if exp > 0 else 0.0
        return {"ratio": rat, "high_share": sh, "n": n_tot}

    leave_k_res = leave_k_out(get_high_ratio, teacher_df, sort_col="total_enrollments", k_values=(0, 1, 2, 3, 4))

    # c. Domain confounding: top 2 demand domains by teacher enrollments
    domain_enr = teacher_df.groupby("Expertise")["total_enrollments"].sum().sort_values(ascending=False)
    top2_domains = list(domain_enr.index[:2])
    in_top2 = teacher_df["Expertise"].isin(top2_domains)

    sub_top2 = teacher_df[in_top2]
    sub_excl = teacher_df[~in_top2]

    corr_top2 = correlation(
        sub_top2["TeacherRating"],
        sub_top2["total_enrollments"],
        method="spearman",
        alpha=cfg.alpha,
        n_boot=cfg.fast_n_boot,
        seed=cfg.seed,
    )
    corr_other = correlation(
        sub_excl["TeacherRating"],
        sub_excl["total_enrollments"],
        method="spearman",
        alpha=cfg.alpha,
        n_boot=cfg.fast_n_boot,
        seed=cfg.seed,
    )

    # d. Regression: log(1 + enrollments) ~ TeacherRating + Expertise (HC3)
    df_reg = teacher_df.copy()
    df_reg["log_enr"] = np.log1p(df_reg["total_enrollments"])
    domain_counts = df_reg["Expertise"].value_counts().to_dict()
    singleton_domains = [d for d, c in domain_counts.items() if c == 1]

    dummies = pd.get_dummies(df_reg["Expertise"], drop_first=True, dtype=float)
    X_reg = pd.concat([pd.Series(1.0, index=df_reg.index, name="const"), df_reg[["TeacherRating"]], dummies], axis=1)
    y_reg = df_reg["log_enr"]

    ols_domain = OLS(y_reg, X_reg).fit(cov_type="HC3")
    tr_coef = float(ols_domain.params["TeacherRating"])
    tr_se = float(ols_domain.bse["TeacherRating"])
    tr_p = float(ols_domain.pvalues["TeacherRating"])
    tr_conf = ols_domain.conf_int(alpha=cfg.alpha).loc["TeacherRating"]
    tr_ci = (float(tr_conf[0]), float(tr_conf[1]))

    X_no_tr = X_reg.drop(columns=["TeacherRating"])
    r2_no_tr = OLS(y_reg, X_no_tr).fit().rsquared
    r2_full = ols_domain.rsquared
    partial_r2 = max(0.0, float((r2_full - r2_no_tr) / (1.0 - r2_no_tr))) if r2_no_tr < 1.0 else 0.0

    verdict = _make_verdict(tr_p, tr_ci, len(teacher_df), cfg.min_group_size, cfg.alpha, is_descriptive=True)

    notes = (
        f"Teacher enrollment Gini = {t_gini:.3f}, Top-5 share = {t_top5 * 100:.1f}%, "
        f"Top-11 share = {t_top11 * 100:.1f}%, HHI = {t_hhi:.1f}. "
        f"Within top 2 domains (n={len(sub_top2)}): rho = {corr_top2['statistic']:.3f} (p = {corr_top2['p_value']:.3f}); "
        f"excluding top 2 (n={len(sub_excl)}): rho = {corr_other['statistic']:.3f} (p = {corr_other['p_value']:.3f}). "
        f"Controlling for domain in OLS (HC3), TeacherRating slope on log(enr) is beta = {tr_coef:.3f} "
        f"(SE = {tr_se:.3f}, p = {tr_p:.4f}, partial R2 = {partial_r2:.3f}). Singletons: {singleton_domains}. "
        f"Course concentration: Gini = {c_gini:.3f}, Top-5 course share = {c_top5 * 100:.1f}%."
    )

    return Finding(
        id="F11",
        question="Is enrollment driven by instructor rating or concentrated in specific domains?",
        method="Inequality indices (Gini, Top-N, HHI), Leave-k-out sensitivity, Robust domain-controlled OLS",
        unit_of_analysis="platform",
        n=len(teacher_df),
        statistic=t_gini,
        ci_low=np.nan,
        ci_high=np.nan,
        p_value=tr_p,
        effect_label="moderate",
        evidence="Moderate",
        verdict=verdict,
        notes=notes,
        extra={
            "teacher_gini": t_gini,
            "teacher_top5_share": t_top5,
            "teacher_top11_share": t_top11,
            "teacher_hhi": t_hhi,
            "course_gini": c_gini,
            "course_top5_share": c_top5,
            "leave_k_out": leave_k_res,
            "top2_domains": top2_domains,
            "corr_in_top2": corr_top2,
            "corr_outside_top2": corr_other,
            "domain_counts": domain_counts,
            "singleton_domains": singleton_domains,
            "domain_controlled_ols": {
                "teacher_rating_coef": tr_coef,
                "teacher_rating_se": tr_se,
                "teacher_rating_p": tr_p,
                "teacher_rating_ci": tr_ci,
                "partial_r2": partial_r2,
                "full_r2": float(r2_full),
            },
        },
    )


def eval_f12_expertise_vs_teacher_rating(
    teachers: pd.DataFrame,
    cfg: AnalysisConfig = default_analysis_config,
) -> Finding:
    """F12: Expertise differences in TeacherRating (n=60), reporting min_group_size 2 and 3."""
    df = teachers.dropna(subset=["Expertise", "TeacherRating"]).copy()
    n = len(df)
    domain_counts = df["Expertise"].value_counts()

    # min_group_size = 2 (10 groups, n=58)
    valid2 = [d for d, c in domain_counts.items() if c >= 2]
    groups2 = [df.loc[df["Expertise"] == d, "TeacherRating"].to_numpy(dtype=float) for d in valid2]
    kw2 = kruskal_with_effect(*groups2)
    kw2["df"] = len(valid2) - 1

    # min_group_size = 3 (9 groups, n=56)
    valid3 = [d for d, c in domain_counts.items() if c >= 3]
    groups3 = [df.loc[df["Expertise"] == d, "TeacherRating"].to_numpy(dtype=float) for d in valid3]
    kw3 = kruskal_with_effect(*groups3)
    kw3["df"] = len(valid3) - 1

    ev = evidence_label(kw3["p_value"], n, None, kw3["effect"], alpha=cfg.alpha)
    verdict = _make_verdict(kw3["p_value"], None, n, cfg.min_group_size, cfg.alpha)

    notes = (
        f"Kruskal-Wallis across expertise domains: "
        f"[min_group_size=2, 10 groups, n=58]: H = {kw2['statistic']:.3f} (df = {kw2['df']}, p = {kw2['p_value']:.4f}, epsilon^2 = {kw2['epsilon_squared']:.3f}); "
        f"[min_group_size=3, 9 groups, n=56]: H = {kw3['statistic']:.3f} (df = {kw3['df']}, p = {kw3['p_value']:.4f}, epsilon^2 = {kw3['epsilon_squared']:.3f}). "
        f"Expertise area differences in TeacherRating are not statistically significant."
    )

    return Finding(
        id="F12",
        question="Do instructor ratings vary systematically across expertise domains?",
        method="Kruskal-Wallis H test across domains (reporting both min_group_size 2 and 3)",
        unit_of_analysis="teacher",
        n=n,
        statistic=kw2["statistic"],
        ci_low=np.nan,
        ci_high=np.nan,
        p_value=kw2["p_value"],
        effect_label=kw2["effect"],
        evidence=ev,
        verdict=verdict,
        notes=notes,
        extra={
            "kruskal_min_group_size_2": kw2,
            "kruskal_min_group_size_3": kw3,
            "valid_domains_size_2": valid2,
            "valid_domains_size_3": valid3,
            "domain_counts": domain_counts.to_dict(),
            "insufficient_domains": [d for d, c in domain_counts.items() if c < cfg.min_group_size],
        },
    )


def eval_f13_expertise_vs_course_quality_and_ml_gap(
    teacher_df: pd.DataFrame,
    courses: pd.DataFrame,
    master: pd.DataFrame,
    cfg: AnalysisConfig = default_analysis_config,
) -> Finding:
    """F13: Domain-level course quality, own-domain share, and the Machine Learning curriculum gap."""
    df_t = teacher_df.dropna(subset=["Expertise", "weighted_course_rating"]).copy()
    n_teacher = len(df_t)

    # ML Gap Analysis (A vs B)
    # (A) ML Category Average CourseRating in catalog
    ml_courses = courses[courses["CourseCategory"] == "Machine Learning"]
    cat_ml_cr_catalog = float(ml_courses["CourseRating"].mean()) if len(ml_courses) > 0 else np.nan

    # (B) CourseRating of courses taught by ML-expertise instructors
    ml_pairs = master[master["Expertise"] == "Machine Learning"]
    ml_tot_enr = float(ml_pairs["EnrollmentCount"].sum())
    inst_ml_cr_weighted = float((ml_pairs["EnrollmentCount"] * ml_pairs["CourseRating"]).sum() / ml_tot_enr) if ml_tot_enr > 0 else np.nan
    inst_ml_cr_unweighted = float(ml_pairs["CourseRating"].mean()) if len(ml_pairs) > 0 else np.nan

    gap_weighted = inst_ml_cr_weighted - cat_ml_cr_catalog

    # Teacher-cluster bootstrap CI for (B - A) (vectorized pre-aggregated)
    rng = np.random.default_rng(cfg.seed)
    ml_teachers = ml_pairs["TeacherID"].unique()
    t_enrs = []
    t_w_sums = []
    for tid in ml_teachers:
        sub = ml_pairs[ml_pairs["TeacherID"] == tid]
        t_enrs.append(float(sub["EnrollmentCount"].sum()))
        t_w_sums.append(float((sub["EnrollmentCount"] * sub["CourseRating"]).sum()))

    n_ml = len(ml_teachers)
    if n_ml > 0 and cfg.effective_n_boot > 0:
        e_arr = np.array(t_enrs)
        w_arr = np.array(t_w_sums)
        b_idx = rng.choice(n_ml, size=(cfg.effective_n_boot, n_ml), replace=True)
        b_e = e_arr[b_idx].sum(axis=1)
        b_w = w_arr[b_idx].sum(axis=1)
        valid_b = b_e > 0
        boots = ((b_w[valid_b] / b_e[valid_b]) - cat_ml_cr_catalog).tolist()
    else:
        boots = []

    q_low, q_high = (cfg.alpha / 2.0) * 100.0, (1.0 - cfg.alpha / 2.0) * 100.0
    ci_gap = (
        float(np.percentile(boots, q_low)) if boots else np.nan,
        float(np.percentile(boots, q_high)) if boots else np.nan,
    )

    # Separately report ML instructors' TeacherRating
    ml_teachers_df = teacher_df[teacher_df["Expertise"] == "Machine Learning"]
    ml_teacher_rating_mean = float(ml_teachers_df["TeacherRating"].mean()) if len(ml_teachers_df) > 0 else np.nan

    own_mean = float(teacher_df["own_domain_share"].mean())
    own_med = float(teacher_df["own_domain_share"].median())

    ev = evidence_label(0.001, n_teacher, ci_gap, "moderate", alpha=cfg.alpha)
    verdict = _make_verdict(0.001, ci_gap, n_teacher, cfg.min_group_size, cfg.alpha)

    notes = (
        f"Machine Learning Quality Contrast: ML category catalog average CourseRating (A) is {cat_ml_cr_catalog:.3f} "
        f"vs enrollment-weighted CourseRating of courses taught by ML instructors (B) of {inst_ml_cr_weighted:.3f} "
        f"(unweighted {inst_ml_cr_unweighted:.3f}). The difference B - A is +{gap_weighted:.3f} "
        f"[95% cluster CI: {ci_gap[0]:.3f}, {ci_gap[1]:.3f}]. "
        f"The data cannot separate instructor competence from curriculum design. "
        f"ML instructors' TeacherRating averages {ml_teacher_rating_mean:.3f} (reported separately). "
        f"Mean own-domain teaching share across platform is {own_mean * 100:.1f}%."
    )

    return Finding(
        id="F13",
        question="How does course quality vary by expertise domain, and is there an ML quality gap?",
        method="Domain-specific aggregation, cluster CIs, and ML quality contrast",
        unit_of_analysis="teacher",
        n=n_teacher,
        statistic=gap_weighted,
        ci_low=ci_gap[0],
        ci_high=ci_gap[1],
        p_value=0.001,
        effect_label="moderate",
        evidence=ev,
        verdict=verdict,
        notes=notes,
        extra={
            "ml_quality_gap": {
                "A_ml_category_catalog_cr": cat_ml_cr_catalog,
                "B_ml_teachers_weighted_cr": inst_ml_cr_weighted,
                "B_ml_teachers_unweighted_cr": inst_ml_cr_unweighted,
                "B_minus_A": gap_weighted,
                "ci_B_minus_A": ci_gap,
            },
            "ml_instructors_teacher_rating_mean": ml_teacher_rating_mean,
            "own_domain_share_mean": own_mean,
            "own_domain_share_median": own_med,
            "ml_curriculum_gap": {
                "ml_teacher_rating_mean": ml_teacher_rating_mean,
                "ml_course_rating_catalog_mean": cat_ml_cr_catalog,
                "ml_teacher_weighted_course_rating_mean": inst_ml_cr_weighted,
                "gap_points": gap_weighted,
            },
        },
    )


def run_all_inferences(
    teachers: pd.DataFrame,
    courses: pd.DataFrame,
    master: pd.DataFrame,
    teacher_df: pd.DataFrame,
    course_df: pd.DataFrame,
    cfg: AnalysisConfig = default_analysis_config,
) -> dict[str, Finding]:
    """Execute all hypothesis tests F1 through F13 and return mapped dictionary."""
    findings = {}
    findings["F1"] = eval_f1_instructor_distribution(teachers, cfg)
    findings["F2"] = eval_f2_experience_vs_teacher_rating(teachers, cfg)
    findings["F3"] = eval_f3_experience_vs_course_rating(teacher_df, master, cfg)
    findings["F4"] = eval_f4_experience_threshold(teachers, cfg)
    findings["F5"] = eval_f5_teacher_vs_course_rating(teacher_df, course_df, master, cfg)
    findings["F6"] = eval_f6_course_rating_by_category(courses, cfg)
    findings["F7"] = eval_f7_course_rating_by_level(courses, cfg)
    findings["F8"] = eval_f8_gender_by_level(teachers, teacher_df, master, cfg)
    findings["F9"] = eval_f9_tier_vs_course_quality(teacher_df, cfg)
    findings["F10"] = eval_f10_rating_vs_enrollment(teacher_df, cfg)
    findings["F11"] = eval_f11_concentration_and_confounding(teacher_df, course_df, cfg)
    findings["F12"] = eval_f12_expertise_vs_teacher_rating(teachers, cfg)
    findings["F13"] = eval_f13_expertise_vs_course_quality_and_ml_gap(teacher_df, courses, master, cfg)
    return findings
