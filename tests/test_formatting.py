"""
tests.test_formatting
=====================
Tests asserting Decimal ROUND_HALF_UP rounding behavior for all formatters.
"""

from __future__ import annotations

import numpy as np

from edupro.formatting import fmt_ci, fmt_half_up, fmt_int, fmt_p_value, fmt_percent, format_slope


def test_fmt_half_up_exact_ties():
    """Verify that ties .005 and .025 round UP, not even."""
    # 3.125 with 2 decimals -> 3.13 (Python round(3.125, 2) is 3.12)
    assert fmt_half_up(3.125, 2) == "3.13"
    assert fmt_half_up(3.1250000000000, 2) == "3.13"
    assert fmt_half_up(3.0978, 2) == "3.10"
    assert fmt_half_up(0.635, 2) == "0.64"
    assert fmt_half_up(3.72, 2) == "3.72"
    assert fmt_half_up(0.120, 3, add_sign=True) == "+0.120"
    assert fmt_half_up(-0.0035, 3, add_sign=True) == "-0.004"


def test_fmt_percent():
    assert fmt_percent(0.4111, 1) == "41.1%"
    assert fmt_percent(0.73957, 1) == "74.0%"
    assert fmt_percent(0.682, 1) == "68.2%"
    assert fmt_percent(0.7312, 1) == "73.1%"


def test_fmt_int():
    assert fmt_int(6182) == "6,182"
    assert fmt_int(10000) == "10,000"
    assert fmt_int(6820.0) == "6,820"


def test_fmt_ci():
    assert fmt_ci(1.033, 1.702, 2) == "[1.03, 1.70]"
    assert fmt_ci(-0.066, 0.418, 3) == "[-0.066, 0.418]"


def test_fmt_p_value():
    assert fmt_p_value(0.0583, 4) == "0.0583"
    assert fmt_p_value(0.00005, 4) == "< 0.0001"
    assert fmt_p_value(np.nan) == "N/A"


def test_format_slope_canonical():
    """format_slope is defined in edupro.formatting and importable from there."""
    assert format_slope(0.120444) == "+0.120"
    assert format_slope(-0.003509) == "-0.0035"
    assert format_slope(0.0001) == "+0.0001"
    assert format_slope(-0.0591) == "-0.059"
    assert format_slope(None) == "N/A"
    assert format_slope(float("nan")) == "N/A"
    assert format_slope(0.120, signed=False) == "0.120"
    assert format_slope(-0.0035, signed=False) == "-0.0035"
