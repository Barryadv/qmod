from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from qmod.config import AppConfig
from qmod.dto import IndicatorDTO, SignalDTO
from qmod.patterns import moving_average_crossover, pattern_to_score

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

np.random.seed(42)


def _get_value(snapshot: List[IndicatorDTO], name: str, key: str = "value") -> Optional[float]:
    for ind in snapshot:
        if ind.name == name and key in ind.values:
            try:
                return float(ind.values[key])
            except Exception:
                return None
    return None


def _rule_score(df: pd.DataFrame, indicators_snapshot: List[IndicatorDTO], config: AppConfig) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    score = 0.0

    rsi_val = _get_value(indicators_snapshot, "rsi")
    if rsi_val is not None:
        if rsi_val < 30:
            score += 0.35
            reasons.append(f"RSI ({rsi_val:.1f}) < 30 ⇒ bullish")
        elif rsi_val > 70:
            score -= 0.35
            reasons.append(f"RSI ({rsi_val:.1f}) > 70 ⇒ bearish")

    macd_hist = _get_value(indicators_snapshot, "macd", key="hist")
    if macd_hist is not None:
        score += 0.15 * np.tanh(macd_hist)
        reasons.append(f"MACD hist {macd_hist:.4f} contribution")

    bb_lower = _get_value(indicators_snapshot, "bollinger", key="lower")
    bb_upper = _get_value(indicators_snapshot, "bollinger", key="upper")
    last_close = float(df["Close"].iloc[-1])
    if bb_lower is not None and last_close < bb_lower:
        score += 0.25
        reasons.append("Close < lower Bollinger ⇒ bullish")
    elif bb_upper is not None and last_close > bb_upper:
        score -= 0.25
        reasons.append("Close > upper Bollinger ⇒ bearish")

    fast = min(config.ti_params.sma_windows) if config.ti_params.sma_windows else 20
    slow = max(config.ti_params.sma_windows) if len(config.ti_params.sma_windows) > 1 else fast * 2
    label, _ = moving_average_crossover(df["Close"], fast, slow)
    pat_score = pattern_to_score(label)
    score += pat_score
    if label != "none":
        reasons.append(f"MA crossover {label} ({pat_score:+.2f})")

    score = float(max(-1.0, min(1.0, score)))
    return score, reasons


def generate_signal(
    df: pd.DataFrame,
    indicators_snapshot: List[IndicatorDTO],
    patterns_snapshot: Any,
    model_or_none: Any,
    config: AppConfig,
) -> SignalDTO:
    score, reasons = _rule_score(df, indicators_snapshot, config)
    sig = "buy" if score > 0.1 else "sell" if score < -0.1 else "hold"
    return SignalDTO(
        date=df.index[-1].to_pydatetime(),
        ticker=config.tkr,
        signal=sig,  # type: ignore
        score=score,
        reasons=reasons,
    )


__all__ = ["generate_signal"]
