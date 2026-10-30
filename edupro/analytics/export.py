"""
edupro.analytics.export
=======================
CLI export tool for EduPro Analytics results.

Executes full analytics pipeline and exports findings, revised KPIs,
entity summaries, and metadata to results/analysis_summary.json.

Usage:
    python -m edupro.analytics.export
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from edupro.analytics.engine import AnalyticsEngine
from edupro.config import PROJECT_ROOT, AnalysisConfig
from edupro.data.pipeline import load_all
from edupro.logging_setup import get_logger

logger = get_logger("analytics.export")


class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder handling numpy types, NaN, and dataclasses."""

    def default(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            if not np.isfinite(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        return super().default(obj)


def run_export(
    output_path: Path | str | None = None,
    fast_mode: bool = False,
) -> Path:
    """Execute full analysis and export summary JSON to disk."""
    if output_path is None:
        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        target_path = results_dir / "analysis_summary.json"
    else:
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset for analysis export...")
    dataset = load_all()

    engine = AnalyticsEngine(
        master=dataset.master,
        teachers_df=dataset.teachers,
        courses_df=dataset.courses,
        transactions_df=dataset.transactions,
    )

    cfg = AnalysisConfig(fast_mode=fast_mode)
    logger.info(f"Running full analysis (fast_mode={fast_mode})...")
    results = engine.run_full_analysis(config=cfg)

    logger.info("Running sensitivity analysis...")
    sensitivity_df = engine.run_sensitivity()

    export_payload = {
        "metadata": results["meta"],
        "kpis_v2": results["kpis_v2"],
        "findings": results["findings"],
        "sensitivity": sensitivity_df.to_dict(orient="records"),
        "entity_tables_preview": {
            "teachers_head": results["entity_tables"]["teachers"].head(5).to_dict(orient="records"),
            "courses_head": results["entity_tables"]["courses"].head(5).to_dict(orient="records"),
        },
    }

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, cls=SafeJSONEncoder, indent=2)

    logger.info(f"Export successfully saved to {target_path}")
    return target_path


if __name__ == "__main__":
    out_file = run_export()
    print(f"\n[EduPro Analytics] Analysis summary successfully exported to {out_file}\n")
