from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import List

import pandas as pd
import yfinance as yf

from qmod.config import AppConfig
from qmod.dto import PriceBarDTO
from qmod.benchmarks import pick_benchmark

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def _ensure_utc_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DatetimeIndex is UTC-based but tz-naive (represents UTC)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex on OHLCV DataFrame.")
    if df.index.tz is not None:
        # Convert to UTC, then drop tz info
        df.index = df.index.tz_convert(timezone.utc).tz_localize(None)
    # else: already tz-naive; treat as UTC by convention
    return df


def fetch_ohlcv(config: AppConfig) -> pd.DataFrame:
    tkr = config.tkr
    logger.info("Downloading OHLCV for %s", tkr)
    df = yf.download(
        tkr,
        start=config.start.isoformat() if config.start else None,
        end=config.end.isoformat() if config.end else None,
        interval=config.interval,
        progress=False,
        auto_adjust=False,   # use raw OHLCV; we will use 'Close' (not Adj Close)
        actions=False,       # skip dividends/splits columns
        # group_by="column", # optional: would avoid MultiIndex entirely
    )

    if df.empty:
        raise ValueError(f"No data returned for ticker {tkr}.")

    # Normalize MultiIndex columns like ('Open','AAPL') -> 'Open'
    if isinstance(df.columns, pd.MultiIndex):
        tkr_key = tkr.upper()
        if tkr_key in df.columns.get_level_values(-1):
            df = df.xs(tkr_key, axis=1, level=-1).copy()
        else:
            df = df.droplevel(-1, axis=1).copy()

    expected_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected OHLCV columns: {missing}")

    df = df[expected_cols].copy()

    # Keep UTC-naive DatetimeIndex and clean
    df = _ensure_utc_naive_index(df)
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(how="all")
    if df.isna().all(axis=None):
        raise ValueError("All OHLCV values are NaN after cleaning.")

    return df

def _yf_download_close(tkr: str, start: datetime, end: datetime) -> pd.Series:
    """Download a single 'Close' series; robust to MultiIndex/1-col DataFrame returns."""
    df = yf.download(tkr, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        logger.warning("No data returned for %s in %s→%s", tkr, start.date(), end.date())
        return pd.Series(dtype="float64", name=tkr)

    s = df.get("Close")
    if s is None:
        logger.warning("'Close' column not found for %s; cols=%s", tkr, list(df.columns))
        return pd.Series(dtype="float64", name=tkr)

    # If yfinance gave a DataFrame (e.g., MultiIndex columns), squeeze to Series
    if isinstance(s, pd.DataFrame):
        s = s.loc[:, ~s.isna().all()].iloc[:, 0] if s.shape[1] else s.squeeze(axis=1)

    s = s.copy()
    s.name = tkr  # temporary; will be overwritten to 'Close'/'BenchmarkClose'
    return s


def fetch_with_benchmark(
    cfg: AppConfig,
    warmup_days: int = 60,
) -> Tuple[pd.DataFrame, str]:
    """
    Fetch primary ticker OHLCV (existing path) PLUS a benchmark close series.
    Align both on the union of trading days (no weekends), forward-fill missing
    values, compute 50D SMA for the ticker, and compute a ratio rebased to 100
    at the first available date >= cfg.start (or earliest possible if needed).

    Returns:
      (df_display, bench_tkr)
      df_display columns:
        - Close           (ticker close)
        - BenchmarkClose  (benchmark close)
        - SMA_50          (50-day SMA of ticker close; needs warm-up)
        - RatioRebased    (ticker/benchmark * 100, rebased at first display date)
    """
    # 1) Resolve date window (with warmup)
    start_dt = pd.to_datetime(cfg.start).to_pydatetime() if cfg.start else None
    end_dt = pd.to_datetime(cfg.end).to_pydatetime() if cfg.end else None
    if start_dt is None or end_dt is None:
        raise ValueError("AppConfig must provide 'start' and 'end' ISO dates.")
    eff_start = start_dt - timedelta(days=warmup_days)
    eff_end = end_dt  # yfinance end is exclusive-ish; leaving as-is is fine

    # 2) Choose benchmark
    _, bench_tkr = pick_benchmark(cfg.tkr)

    logger.info("Downloading OHLCV for %s (warmup %dd) and benchmark %s",
                cfg.tkr, warmup_days, bench_tkr)

    # 3) Download both closes
    t_close = _yf_download_close(cfg.tkr, eff_start, eff_end)
    b_close = _yf_download_close(bench_tkr, eff_start, eff_end)

    if t_close.empty:
        raise ValueError(f"No price data for {cfg.tkr} between {eff_start.date()}–{eff_end.date()}.")

    if b_close.empty:
        logger.warning("No benchmark data for %s; fallback to ratio=NaN", bench_tkr)

    # 4) Align on union of trading days (excludes weekends/holidays automatically)
    # Outer-join, then ffill to handle days when one market traded and the other didn't.


    t_close.name = "Close"
    b_close.name = "BenchmarkClose"
    aligned = pd.concat([t_close, b_close], axis=1, join="outer").sort_index()

    # Forward-fill each independently; we do NOT insert weekends/holidays.
    aligned = aligned.ffill()

    # 5) Compute SMA_50 (need warm-up)
    sma_50 = aligned["Close"].rolling(window=50, min_periods=1).mean()
    aligned["SMA_50"] = sma_50

    # 6) Compute raw ratio and rebase to 100 at first usable display date
    # Display should begin at max(cfg.start, first valid close for both)
    first_valid_close = aligned["Close"].dropna().index.min()
    first_valid_bench = aligned["BenchmarkClose"].dropna().index.min()
    display_start = max(
        dt for dt in [start_dt, first_valid_close, first_valid_bench] if dt is not None
    )
    if display_start > end_dt:
        raise ValueError(
            f"No overlapping data for {cfg.tkr} and {bench_tkr} in requested range."
        )

    # Ratio
    ratio = aligned["Close"] / aligned["BenchmarkClose"]
    # Rebase at first display date (fall back to first available if needed)
    try:
        base = ratio.loc[ratio.index >= display_start].iloc[0]
    except IndexError:
        base = ratio.dropna().iloc[0]
        display_start = ratio.dropna().index[0]
        logger.warning(
            "Adjusted display start to %s due to missing data on requested start.",
            display_start.date()
        )
    aligned["RatioRebased"] = (ratio / base) * 100.0

    # 7) Trim to display range
    df_display = aligned.loc[(aligned.index >= display_start) & (aligned.index <= end_dt)].copy()

    # Safety: drop rows with both NaNs (shouldn’t occur after ffill, but keep clean)
    df_display = df_display.dropna(how="all")

    return df_display, bench_tkr

def bars_from_df(df: pd.DataFrame, tkr: str) -> List[PriceBarDTO]:
    """Build PriceBarDTOs from an OHLCV frame; tolerates NaN volume."""
    required = {"Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    bars: List[PriceBarDTO] = []
    for ts, row in df.iterrows():
        t: datetime = ts.to_pydatetime()
        bars.append(
            PriceBarDTO(
                t=t,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=None if ("Volume" not in df.columns or pd.isna(row.get("Volume"))) else float(row["Volume"]),
            )
        )
    return bars


__all__ = ["fetch_ohlcv", "bars_from_df"]
