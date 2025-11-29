from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from qmod.config import AppConfig, TechnicalParams
from qmod.dto import IndicatorDTO

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=series.index).rolling(window).mean()
    roll_down = pd.Series(loss, index=series.index).rolling(window).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    bandwidth = (upper - lower) / mid
    percent_b = (series - lower) / (upper - lower)
    return pd.DataFrame(
        {"mid": mid, "upper": upper, "lower": lower, "bandwidth": bandwidth, "percent_b": percent_b}
    )


def compute_indicator_snapshot(df: pd.DataFrame, config: AppConfig) -> List[IndicatorDTO]:
    t: TechnicalParams = config.ti_params
    close = df["Close"]

    # --- ensure 1D numeric series ---
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce")

    indicators: List[IndicatorDTO] = []

    # SMA
    for w in t.sma_windows:
        s = sma(close, w)
        last = s.iloc[-1]
        if pd.notna(last):
            indicators.append(
                IndicatorDTO(name=f"sma_{w}", params={"window": w}, values={"value": float(last)})
            )

    # EMA
    for w in t.ema_windows:
        e = ema(close, w)
        last = e.iloc[-1]
        if pd.notna(last):
            indicators.append(
                IndicatorDTO(name=f"ema_{w}", params={"span": w}, values={"value": float(last)})
            )

    # RSI
    r = rsi(close, t.rsi_window)
    last_rsi = r.iloc[-1]
    if pd.notna(last_rsi):
        indicators.append(
            IndicatorDTO(name="rsi", params={"window": t.rsi_window}, values={"value": float(last_rsi)})
        )

    # MACD
    m = macd(close, t.macd_fast, t.macd_slow, t.macd_signal)
    macd_last = m["macd"].iloc[-1]
    sig_last = m["signal"].iloc[-1]
    hist_last = m["hist"].iloc[-1]
    if pd.notna(macd_last) and pd.notna(sig_last) and pd.notna(hist_last):
        indicators.append(
            IndicatorDTO(
                name="macd",
                params={"fast": t.macd_fast, "slow": t.macd_slow, "signal": t.macd_signal},
                values={"macd": float(macd_last), "signal": float(sig_last), "hist": float(hist_last)},
            )
        )

    # Bollinger
    bb = bollinger(close, t.bb_window, t.bb_std)
    mid_last = bb["mid"].iloc[-1]
    if pd.notna(mid_last):
        indicators.append(
            IndicatorDTO(
                name="bollinger",
                params={"window": t.bb_window, "std": t.bb_std},
                values={
                    "mid": float(mid_last),
                    "upper": float(bb["upper"].iloc[-1]),
                    "lower": float(bb["lower"].iloc[-1]),
                    "percent_b": float(bb["percent_b"].iloc[-1]),
                },
            )
        )

    return indicators



__all__ = ["sma", "ema", "rsi", "macd", "bollinger", "compute_indicator_snapshot"]
