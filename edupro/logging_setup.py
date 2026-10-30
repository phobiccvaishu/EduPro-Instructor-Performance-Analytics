"""
edupro.logging_setup
====================
Logging configuration for EduPro.
"""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with consistent formatting if not already configured."""
    root = logging.getLogger("edupro")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
        root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a logger instance under the edupro namespace."""
    setup_logging()
    if not name.startswith("edupro.") and name != "edupro":
        name = f"edupro.{name}"
    return logging.getLogger(name)
