"""
edupro.analytics.stats
======================
Pure statistical helper functions for EduPro Analytics.

Includes:
- Robust correlation with Fisher-z (Pearson) and bootstrap (Spearman) CIs
- Standard and cluster-aware bootstrap confidence intervals
- Nonparametric tests with effect sizes (Kruskal-Wallis, Mann-Whitney U)
- Multiple testing correction (Holm-Bonferroni)
- Inequality and concentration metrics (Gini, Top-N share, HHI)
- Sensitivity analysis (leave-k-out)
- Structured evidence labeling
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from edupro.config import (
    get_correlation_effect,
    get_epsilon_squared_effect,
    get_rank_biserial_effect,
)


def correlation(
    x: Sequence[float] | np.ndarray | pd.Series,
    y: Sequence[float] | np.ndarray | pd.Series,
    method: str = "pearson",
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int | None = 20260601,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """
    Compute Pearson or Spearman correlation with p-value, sample size,
    95% CI (Fisher-z for Pearson, bootstrap for Spearman), and effect size label.

    Returns dict with status="insufficient data" when n < 3 or variance is zero.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    # Filter pairwise finite values
    valid_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_clean = x_arr[valid_mask]
    y_clean = y_arr[valid_mask]
    n = len(x_clean)

    empty_res = {
        "status": "insufficient data",
        "method": method,
        "n": n,
        "statistic": np.nan,
        "p_value": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "effect": "undefined",
    }

    if n < 3:
        return empty_res

    # Check for zero variance / constant input
    if np.all(x_clean == x_clean[0]) or np.all(y_clean == y_clean[0]):
        return empty_res

    method_lower = method.lower()
    if method_lower == "pearson":
        res = stats.pearsonr(x_clean, y_clean)
        r = float(res.statistic)
        p_val = float(res.pvalue)

        # Fisher-z transformation for Pearson 95% CI
        # Clip r slightly away from +/- 1.0 to avoid arctanh infinity
        r_clipped = np.clip(r, -0.999999, 0.999999)
        z = np.arctanh(r_clipped)
        se = 1.0 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
        ci_low = float(np.tanh(z - z_crit * se))
        ci_high = float(np.tanh(z + z_crit * se))

        effect = get_correlation_effect(r)
        return {
            "status": "ok",
            "method": "pearson",
            "n": n,
            "statistic": r,
            "p_value": p_val,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "effect": effect,
        }

    elif method_lower == "spearman":
        res = stats.spearmanr(x_clean, y_clean)
        rho = float(res.statistic)
        p_val = float(res.pvalue)

        # Bootstrap percentile CI for Spearman
        if rng is None:
            rng = np.random.default_rng(seed)

        boot_stats = []
        indices = np.arange(n)
        for _ in range(n_boot):
            boot_idx = rng.choice(indices, size=n, replace=True)
            bx, by = x_clean[boot_idx], y_clean[boot_idx]
            if np.all(bx == bx[0]) or np.all(by == by[0]):
                continue
            b_res = stats.spearmanr(bx, by)
            if np.isfinite(b_res.statistic):
                boot_stats.append(b_res.statistic)

        if len(boot_stats) < 50:
            ci_low, ci_high = np.nan, np.nan
        else:
            q_low = (alpha / 2.0) * 100.0
            q_high = (1.0 - alpha / 2.0) * 100.0
            ci_low = float(np.percentile(boot_stats, q_low))
            ci_high = float(np.percentile(boot_stats, q_high))

        effect = get_correlation_effect(rho)
        return {
            "status": "ok",
            "method": "spearman",
            "n": n,
            "statistic": rho,
            "p_value": p_val,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "effect": effect,
        }
    else:
        raise ValueError(f"Unsupported correlation method: {method}. Choose 'pearson' or 'spearman'.")


def bootstrap_ci(
    data: Sequence[Any] | np.ndarray | pd.DataFrame | pd.Series,
    stat_fn: Callable[[Any], float] | None = None,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int | None = 20260601,
    rng: np.random.Generator | None = None,
    cluster_ids: Sequence[Any] | np.ndarray | pd.Series | None = None,
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval for a scalar statistic.
    Supports standard i.i.d. resampling or cluster-aware block resampling.

    If cluster_ids is provided, unique clusters are resampled with replacement,
    preserving the within-cluster correlation structure.
    """
    if stat_fn is None:
        def default_mean(x: Any) -> float:
            arr = np.asarray(x, dtype=float)
            return float(np.nanmean(arr))
        stat_fn = default_mean

    if rng is None:
        rng = np.random.default_rng(seed)

    alpha = 1.0 - ci
    q_low = (alpha / 2.0) * 100.0
    q_high = (1.0 - alpha / 2.0) * 100.0

    if cluster_ids is not None:
        # Cluster-aware resampling
        if isinstance(data, (pd.DataFrame, pd.Series)):
            df = data.copy()
            df["_cluster_id"] = np.asarray(cluster_ids)
        else:
            df = pd.DataFrame({"val": data, "_cluster_id": cluster_ids})

        clusters = df["_cluster_id"].unique()
        n_clusters = len(clusters)
        if n_clusters < 2:
            return (np.nan, np.nan)

        boot_estimates = []
        cluster_groups = {cid: grp for cid, grp in df.groupby("_cluster_id")}

        for _ in range(n_boot):
            selected_cids = rng.choice(clusters, size=n_clusters, replace=True)
            boot_sample = pd.concat([cluster_groups[cid] for cid in selected_cids], ignore_index=True)
            try:
                est = stat_fn(boot_sample)
                if np.isfinite(est):
                    boot_estimates.append(est)
            except Exception:
                continue

        if len(boot_estimates) < 30:
            return (np.nan, np.nan)

        return (
            float(np.percentile(boot_estimates, q_low)),
            float(np.percentile(boot_estimates, q_high)),
        )

    else:
        # Standard observation-level resampling
        arr = np.asarray(data)
        n = len(arr)
        if n < 2:
            return (np.nan, np.nan)

        boot_estimates = []
        indices = np.arange(n)
        for _ in range(n_boot):
            boot_idx = rng.choice(indices, size=n, replace=True)
            sample = arr[boot_idx]
            try:
                est = stat_fn(sample)
                if np.isfinite(est):
                    boot_estimates.append(est)
            except Exception:
                continue

        if len(boot_estimates) < 30:
            return (np.nan, np.nan)

        return (
            float(np.percentile(boot_estimates, q_low)),
            float(np.percentile(boot_estimates, q_high)),
        )


def kruskal_with_effect(*groups: Sequence[float] | np.ndarray | pd.Series) -> dict[str, Any]:
    """
    Perform Kruskal-Wallis H-test across k independent groups and calculate
    epsilon-squared effect size (Tomczak & Tomczak 2014):
    epsilon^2 = (H - k + 1) / (n - k) bounded in [0, 1].
    """
    cleaned_groups = []
    for g in groups:
        arr = np.asarray(g, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) > 0:
            cleaned_groups.append(arr)

    k = len(cleaned_groups)
    total_n = sum(len(g) for g in cleaned_groups)

    empty_res = {
        "status": "insufficient data",
        "k": k,
        "n": total_n,
        "statistic": np.nan,
        "p_value": np.nan,
        "epsilon_squared": np.nan,
        "effect": "undefined",
    }

    if k < 2 or total_n < 3:
        return empty_res

    # Check if all observations across all groups are identical
    all_vals = np.concatenate(cleaned_groups)
    if np.all(all_vals == all_vals[0]):
        return empty_res

    res = stats.kruskal(*cleaned_groups)
    h_stat = float(res.statistic)
    p_val = float(res.pvalue)

    # Epsilon-squared formula
    if total_n > k:
        eps_sq = (h_stat - k + 1.0) / (total_n - k)
        eps_sq = float(np.clip(eps_sq, 0.0, 1.0))
    else:
        eps_sq = float(np.clip(h_stat / max(total_n - 1, 1), 0.0, 1.0))

    effect = get_epsilon_squared_effect(eps_sq)

    return {
        "status": "ok",
        "k": k,
        "n": total_n,
        "statistic": h_stat,
        "p_value": p_val,
        "epsilon_squared": eps_sq,
        "effect": effect,
    }


def mann_whitney_with_effect(
    x: Sequence[float] | np.ndarray | pd.Series,
    y: Sequence[float] | np.ndarray | pd.Series,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """
    Compute Mann-Whitney U test between two independent groups and calculate
    rank-biserial correlation: r_rb = 1 - (2*U) / (n1 * n2).
    """
    x_arr = np.asarray(x, dtype=float)
    x_clean = x_arr[np.isfinite(x_arr)]
    y_arr = np.asarray(y, dtype=float)
    y_clean = y_arr[np.isfinite(y_arr)]

    n1 = len(x_clean)
    n2 = len(y_clean)

    empty_res = {
        "status": "insufficient data",
        "n1": n1,
        "n2": n2,
        "statistic": np.nan,
        "p_value": np.nan,
        "rank_biserial": np.nan,
        "effect": "undefined",
    }

    if n1 < 1 or n2 < 1 or (n1 + n2) < 3:
        return empty_res

    if np.all(x_clean == x_clean[0]) and np.all(y_clean == y_clean[0]) and x_clean[0] == y_clean[0]:
        return empty_res

    res = stats.mannwhitneyu(x_clean, y_clean, alternative=alternative)
    u_stat = float(res.statistic)
    p_val = float(res.pvalue)

    # Rank-biserial correlation
    r_rb = 1.0 - (2.0 * u_stat) / (n1 * n2)
    effect = get_rank_biserial_effect(r_rb)

    return {
        "status": "ok",
        "n1": n1,
        "n2": n2,
        "statistic": u_stat,
        "p_value": p_val,
        "rank_biserial": float(r_rb),
        "effect": effect,
    }


def holm_adjust(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Adjust p-values using the step-down Holm-Bonferroni method.
    Preserves original order and handles NaNs.
    """
    p_arr = np.asarray(p_values, dtype=float)
    m = len(p_arr)
    if m == 0:
        return np.array([], dtype=float)

    adjusted = np.full(m, np.nan, dtype=float)
    valid_idx = np.where(np.isfinite(p_arr))[0]
    k = len(valid_idx)
    if k == 0:
        return adjusted

    # Sort valid p-values ascending
    valid_p = p_arr[valid_idx]
    sort_order = np.argsort(valid_p)
    sorted_p = valid_p[sort_order]

    # Step-down adjustment: p_adj_i = max(p_adj_{i-1}, (k - i) * p_i)
    cum_max = 0.0
    adj_sorted = np.zeros(k, dtype=float)
    for i in range(k):
        multiplier = k - i
        val = min(1.0, multiplier * sorted_p[i])
        cum_max = max(cum_max, val)
        adj_sorted[i] = cum_max

    # Put back into original positions
    orig_positions = valid_idx[sort_order]
    adjusted[orig_positions] = adj_sorted
    return adjusted


def gini(values: Sequence[float] | np.ndarray | pd.Series) -> float:
    """
    Compute Gini coefficient of inequality for non-negative values.
    Returns 0.0 for uniform distribution, 1.0 for complete concentration.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 2 or np.all(arr == 0):
        return 0.0

    # Ensure non-negative
    arr = np.sort(np.maximum(arr, 0.0))
    total = np.sum(arr)
    if total == 0:
        return 0.0

    index = np.arange(1, n + 1)
    g = (2.0 * np.sum(index * arr)) / (n * total) - (n + 1.0) / n
    return float(np.clip(g, 0.0, 1.0))


def top_n_share(values: Sequence[float] | np.ndarray | pd.Series, n: int) -> float:
    """
    Compute share of the total sum held by the top n largest observations.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0 or np.sum(arr) <= 0:
        return 0.0
    sorted_desc = np.sort(arr)[::-1]
    top_sum = np.sum(sorted_desc[:n])
    return float(top_sum / np.sum(arr))


def hhi(shares: Sequence[float] | np.ndarray | pd.Series) -> float:
    """
    Calculate Herfindahl-Hirschman Index (HHI).
    Expects fractional shares (summing to 1.0) or counts.
    Returns HHI on standard 0 to 10,000 scale.
    """
    arr = np.asarray(shares, dtype=float)
    arr = arr[np.isfinite(arr)]
    total = np.sum(arr)
    if total <= 0:
        return 0.0
    normalized_pct = (arr / total) * 100.0
    return float(np.sum(normalized_pct ** 2))


def leave_k_out(
    stat_fn: Callable[[pd.DataFrame], Any],
    data: pd.DataFrame,
    sort_col: str,
    k_values: Sequence[int] = (0, 1, 2, 3, 4),
) -> dict[int, Any]:
    """
    Evaluate stat_fn while sequentially removing the top k rows
    ordered descending by sort_col.
    """
    sorted_df = data.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
    results = {}
    for k in k_values:
        if k == 0:
            sub = sorted_df
        elif k >= len(sorted_df):
            sub = sorted_df.iloc[0:0]
        else:
            sub = sorted_df.iloc[k:].reset_index(drop=True)
        results[k] = stat_fn(sub)
    return results


def evidence_label(
    p_value: float | None,
    n: int,
    ci: tuple[float, float] | None,
    effect: str,
    alpha: float = 0.05,
    min_group_size: int = 3,
) -> str:
    """
    Assign qualitative evidence rating:
    - 'Strong': statistically significant, CI excludes zero, moderate/large effect, adequate sample size
    - 'Moderate': statistically significant, CI excludes zero, small effect or smaller sample size
    - 'Weak': borderline significant (p < 0.10) or significant with negligible effect
    - 'None': p >= alpha or CI crosses zero with negligible effect
    - 'Insufficient data': n < min_group_size or invalid inputs
    """
    if n < min_group_size or p_value is None or not np.isfinite(p_value):
        return "Insufficient data"

    ci_excludes_zero = True
    if ci is not None and len(ci) == 2:
        low, high = ci
        if np.isfinite(low) and np.isfinite(high):
            if low <= 0.0 <= high:
                ci_excludes_zero = False

    if p_value < alpha and ci_excludes_zero:
        if effect in ("large", "moderate") and n >= 30:
            return "Strong"
        elif effect in ("large", "moderate"):
            return "Moderate"
        elif effect == "small":
            return "Moderate" if n >= 30 else "Weak"
        else:  # negligible
            return "Weak"
    elif p_value < 0.10:
        return "Weak"
    else:
        return "None"
