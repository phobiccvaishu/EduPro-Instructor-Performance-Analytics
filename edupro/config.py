"""
edupro.config
=============
Central configuration, statistical parameters, and domain constants for EduPro Analytics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Project root: directory containing the app and edupro package
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AnalysisConfig:
    """Runtime parameters for statistical analysis, resampling, and attribution."""

    alpha: float = 0.05
    seed: int = 20260601
    n_boot: int = 2000
    n_perm: int = 2000
    fast_mode: bool = False
    fast_n_boot: int = 300
    fast_n_perm: int = 0
    min_group_size: int = 3
    attribution: str = "enrollment_weighted"  # "enrollment_weighted" or "unweighted"
    min_pair_enrollment: int | None = None  # None for default; e.g. 3 for sensitivity runs
    robustness_retention_threshold: float = 0.75
    robustness_k: int = 3
    fragile_evidence_cap: str = "Weak"
    min_cluster_size_reliable: int = 8
    small_cluster_evidence_cap: str = "Weak"

    @property
    def effective_n_boot(self) -> int:
        """Effective bootstrap iterations respecting fast mode."""
        return self.fast_n_boot if self.fast_mode else self.n_boot

    @property
    def effective_n_perm(self) -> int:
        """Effective permutation iterations respecting fast mode."""
        return self.fast_n_perm if self.fast_mode else self.n_perm


@dataclass(frozen=True)
class Settings:
    """Application configuration and analytical settings."""

    # File paths
    data_path_env_var: str = "EDUPRO_DATA_PATH"
    default_data_rel_path: str = "data/raw/EduPro_Online_Platform.xlsx"

    # Statistical Defaults
    default_alpha: float = 0.05
    default_seed: int = 20260601
    default_n_boot: int = 2000
    default_n_perm: int = 2000
    fast_n_boot: int = 300
    fast_n_perm: int = 0
    attribution_default: str = "enrollment_weighted"
    min_pair_enrollment_sensitivity: int = 3
    robustness_retention_threshold: float = 0.75
    robustness_k: int = 3
    fragile_evidence_cap: str = "Weak"
    min_cluster_size_reliable: int = 8
    small_cluster_evidence_cap: str = "Weak"
    allowed_verdicts: tuple[str, ...] = (
        "supported",
        "not supported",
        "inconclusive",
        "descriptive",
        "insufficient data",
    )

    # Rating scale
    min_rating: float = 1.0
    max_rating: float = 5.0

    # Tier thresholds
    tier_high_threshold: float = 4.0
    tier_low_threshold: float = 3.0

    # Sample size thresholds
    min_group_size: int = 3

    # Age bounds
    min_teacher_age: int = 18
    min_user_age: int = 13
    max_age: int = 100

    # Experience bands
    exp_bins: tuple[int, ...] = (0, 5, 10, 15, 100)
    exp_labels: tuple[str, ...] = ("0–5 yrs", "6–10 yrs", "11–15 yrs", "16+ yrs")

    @property
    def data_path(self) -> Path:
        """Resolve data file path relative to project root or environment variable."""
        raw_env = os.environ.get(self.data_path_env_var)
        if raw_env:
            p = Path(raw_env)
            return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
        return (PROJECT_ROOT / self.default_data_rel_path).resolve()

    @property
    def tier_high_label(self) -> str:
        """Label for high tier instructors (e.g. 'High (≥4.0)')."""
        return f"High (≥{self.tier_high_threshold:.1f})"

    @property
    def tier_mid_label(self) -> str:
        """Label for mid tier instructors (e.g. 'Mid (3.0–3.9)')."""
        return f"Mid ({self.tier_low_threshold:.1f}–{self.tier_high_threshold - 0.1:.1f})"

    @property
    def tier_low_label(self) -> str:
        """Label for low tier instructors (e.g. 'Low (<3.0)')."""
        return f"Low (<{self.tier_low_threshold:.1f})"

    @property
    def tier_labels(self) -> tuple[str, str, str]:
        """Tuple of tier labels in descending quality order."""
        return (self.tier_high_label, self.tier_mid_label, self.tier_low_label)

    def get_tier(self, rating: float | None) -> str:
        """Assign rating to tier label."""
        if rating is None or rating != rating:  # NaN check
            return "Unknown"
        if rating >= self.tier_high_threshold:
            return self.tier_high_label
        elif rating >= self.tier_low_threshold:
            return self.tier_mid_label
        else:
            return self.tier_low_label

    def classify_tier(self, rating: float | None) -> str:
        """Alias for get_tier."""
        return self.get_tier(rating)


def get_correlation_effect(r: float | None) -> str:
    """Classify correlation effect size according to standard Cohen benchmarks."""
    if r is None or r != r:
        return "undefined"
    abs_r = abs(r)
    if abs_r < 0.10:
        return "negligible"
    elif abs_r < 0.30:
        return "small"
    elif abs_r < 0.50:
        return "moderate"
    else:
        return "large"


def get_epsilon_squared_effect(eps: float | None) -> str:
    """Classify Kruskal-Wallis epsilon-squared effect size."""
    if eps is None or eps != eps:
        return "undefined"
    eps_val = max(0.0, float(eps))
    if eps_val < 0.01:
        return "negligible"
    elif eps_val < 0.08:
        return "small"
    elif eps_val < 0.26:
        return "moderate"
    else:
        return "large"


def get_rank_biserial_effect(r_rb: float | None) -> str:
    """Classify Mann-Whitney rank-biserial correlation effect size."""
    if r_rb is None or r_rb != r_rb:
        return "undefined"
    abs_r = abs(r_rb)
    if abs_r < 0.10:
        return "negligible"
    elif abs_r < 0.30:
        return "small"
    elif abs_r < 0.50:
        return "moderate"
    else:
        return "large"


settings = Settings()
default_analysis_config = AnalysisConfig()
