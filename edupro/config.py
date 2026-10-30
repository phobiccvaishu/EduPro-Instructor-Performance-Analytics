"""
edupro.config
=============
Central configuration and domain constants for EduPro Analytics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Project root: directory containing the app and edupro package
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Application configuration and analytical settings."""

    # File paths
    data_path_env_var: str = "EDUPRO_DATA_PATH"
    default_data_rel_path: str = "data/raw/EduPro_Online_Platform.xlsx"

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


settings = Settings()
