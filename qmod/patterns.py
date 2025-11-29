from __future__ import annotations

import logging
from typing import Literal, Tuple

import numpy as np
import pandas as pd

from qmod.indicators import sma

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def moving_average_crossover(
    close: pd.Series, fast_win: int, slow_win: int
) -> Tuple[Literal["bullish", "bearish", "none"], float]:
    """Detect SMA crossover pattern (coerces input to 1-D numeric Series)."""
    # --- Ensure 1-D numeric Series ---
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce")

    if fast_win >= slow_win:
        raise ValueError("fast_win must be < slow_win for crossover.")

    f = sma(close, fast_win)
    s = sma(close, slow_win)
    if len(close) < slow_win + 2:
        return "none", 0.0

    f_prev, f_curr = f.iloc[-2], f.iloc[-1]
    s_prev, s_curr = s.iloc[-2], s.iloc[-1]

    # Guard NaNs safely
    if any(pd.isna(x) for x in (f_prev, f_curr, s_prev, s_curr)):
        return "none", 0.0

    crossed_up = (f_prev <= s_prev) and (f_curr > s_curr)
    crossed_down = (f_prev >= s_prev) and (f_curr < s_curr)

    if crossed_up:
        dist = abs((f_curr - s_curr) / s_curr) if s_curr != 0 else 0.0
        return "bullish", float(max(0.1, min(1.0, 5.0 * dist)))
    if crossed_down:
        dist = abs((f_curr - s_curr) / s_curr) if s_curr != 0 else 0.0
        return "bearish", float(max(0.1, min(1.0, 5.0 * dist)))
    return "none", 0.0


def pattern_to_score(label: Literal["bullish", "bearish", "none"]) -> float:
    """Map pattern label to a small score contribution."""
    if label == "bullish":
        return 0.2
    if label == "bearish":
        return -0.2
    return 0.0


__all__ = ["moving_average_crossover", "pattern_to_score"]
