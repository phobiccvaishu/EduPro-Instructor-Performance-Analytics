"""
edupro.formatting
=================
Centralized numerical formatting utilities enforcing Decimal ROUND_HALF_UP
display semantics across all UI tiles, cards, tables, and insight strings.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any
import math
import numpy as np


def to_decimal(val: Any) -> Decimal | None:
    """Safely convert numeric value or string to Decimal."""
    if val is None:
        return None
    if isinstance(val, Decimal):
        return val
    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return None
        # Convert float to string via repr/str for decimal construction
        return Decimal(str(val))
    if isinstance(val, (int, np.integer)):
        return Decimal(int(val))
    if isinstance(val, str):
        val_str = val.strip().replace(",", "")
        try:
            return Decimal(val_str)
        except Exception:
            return None
    return None


def fmt_half_up(
    val: Any,
    decimals: int = 2,
    prefix: str = "",
    suffix: str = "",
    na_rep: str = "N/A",
    add_sign: bool = False,
) -> str:
    """
    Format numeric value using Decimal ROUND_HALF_UP arithmetic.
    
    Examples:
        fmt_half_up(3.125, 2) -> "3.13"
        fmt_half_up(0.1204, 3, add_sign=True) -> "+0.120"
    """
    dec = to_decimal(val)
    if dec is None:
        return na_rep

    quantum_str = "1" if decimals == 0 else f"0.{'0' * (decimals - 1)}1"
    quantum = Decimal(quantum_str)
    rounded = dec.quantize(quantum, rounding=ROUND_HALF_UP)

    # Format with standard comma grouping if integer part >= 1000
    int_part = abs(int(rounded))
    int_str = f"{int_part:,}"
    if decimals > 0:
        dec_str = str(abs(rounded)).split(".")[1] if "." in str(abs(rounded)) else "0" * decimals
        # Pad dec_str if needed
        dec_str = dec_str.ljust(decimals, "0")
        formatted = f"{int_str}.{dec_str}"
    else:
        formatted = int_str

    if rounded < 0:
        res = f"-{formatted}"
    elif add_sign and rounded > 0:
        res = f"+{formatted}"
    else:
        res = formatted

    return f"{prefix}{res}{suffix}"


def fmt_percent(
    val: Any,
    decimals: int = 1,
    mult_100: bool = True,
    na_rep: str = "N/A",
) -> str:
    """Format fractional share or percentage with ROUND_HALF_UP."""
    dec = to_decimal(val)
    if dec is None:
        return na_rep
    if mult_100:
        dec = dec * Decimal("100")
    return fmt_half_up(dec, decimals=decimals, suffix="%", na_rep=na_rep)


def fmt_int(val: Any, na_rep: str = "N/A") -> str:
    """Format integer with thousands separator."""
    dec = to_decimal(val)
    if dec is None:
        return na_rep
    rounded = dec.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return f"{int(rounded):,}"


def fmt_ci(
    low: Any,
    high: Any,
    decimals: int = 2,
    na_rep: str = "N/A",
    brackets: tuple[str, str] = ("[", "]"),
) -> str:
    """Format confidence interval [low, high] with ROUND_HALF_UP."""
    str_low = fmt_half_up(low, decimals=decimals, na_rep=na_rep)
    str_high = fmt_half_up(high, decimals=decimals, na_rep=na_rep)
    if str_low == na_rep or str_high == na_rep:
        return na_rep
    return f"{brackets[0]}{str_low}, {str_high}{brackets[1]}"


def fmt_p_value(p: Any, decimals: int = 4, na_rep: str = "N/A") -> str:
    """Format p-value with standard small-p threshold notation."""
    dec = to_decimal(p)
    if dec is None:
        return na_rep
    cutoff = Decimal(f"0.{'0' * (decimals - 1)}1")
    if dec < cutoff:
        return f"< 0.{'0' * (decimals - 1)}1"
    return fmt_half_up(dec, decimals=decimals, na_rep=na_rep)
