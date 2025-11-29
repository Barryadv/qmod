from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional, Literal

import numpy as np
import pandas as pd

from qmod.config import AppConfig
from qmod.data_importer import fetch_with_benchmark, fetch_ohlcv
from qmod.dto import ChartDTO, IndicatorDTO, RunOutputDTO
from qmod.indicators import compute_indicator_snapshot, sma
from qmod.patterns import moving_average_crossover, pattern_to_score  # pattern_to_score kept for future use
from qmod.charting import price_with_indicators_chart, relative_vs_price_chart
from qmod.signals import generate_signal
from qmod.paths import artifacts_dir
from qmod.report_generator import save_html_report

# Optuna optimizer modules (lazy import to avoid hard dependency)
_optuna_imp = None
_optuna_imp_rsi = None

try:
    from qmod import optuna_imp as _optuna_imp
    from qmod import optuna_imp_rsi as _optuna_imp_rsi
except ImportError:
    pass  # Optimizers not available

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# For reproducible behavior in any stochastic components (if added later)
np.random.seed(42)

# Default parameters when optimization is skipped
DEFAULT_MACD = {"fast": 12, "slow": 26, "signal": 9}
DEFAULT_RSI = {"length": 14, "os": 30, "ob": 70}

OptimizerMode = Literal["macd", "rsi", "both"]


# ---------------------------------------------------------------------------
# Helper functions for optimization
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _macd_hist(close: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    return macd_line - signal_line


def _rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _safe_tkr(tkr: str) -> str:
    return tkr.replace(":", "_").replace("-", "_").replace("/", "_").strip()


def optimize_macd(
    close: pd.Series,
    tkr: str,
    n_trials: int,
) -> Dict[str, int]:
    """Run MACD optimization and return best parameters."""
    if _optuna_imp is None:
        logger.warning("MACD optimizer not available; using defaults")
        return DEFAULT_MACD.copy()

    def objective(trial):
        fast = trial.suggest_int("fast", 5, 20)
        slow = trial.suggest_int("slow", fast + 4, 50)
        signal = trial.suggest_int("signal", 3, 15)

        hist = _macd_hist(close, fast, slow, signal).dropna()
        if hist.empty:
            return -1e9

        hret = hist.diff().dropna()
        if hret.std() == 0:
            return -1e9

        sharpe = hret.mean() / hret.std() * np.sqrt(252)
        return float(sharpe)

    vis_dir = artifacts_dir("optimized", _safe_tkr(tkr), "macd")
    best_params = _optuna_imp.optimize_with_optuna(
        objective_func=objective,
        vis_dir=vis_dir,
        n_trials=n_trials,
        study_name=f"macd_{tkr}",
        direction="maximize",
    )

    result = DEFAULT_MACD.copy()
    for k in ("fast", "slow", "signal"):
        if k in best_params:
            result[k] = int(best_params[k])

    logger.info("MACD optimized params: %s", result)
    return result


def optimize_rsi(
    close: pd.Series,
    tkr: str,
    n_trials: int,
) -> Dict[str, int]:
    """Run RSI optimization and return best parameters."""
    if _optuna_imp_rsi is None:
        logger.warning("RSI optimizer not available; using defaults")
        return DEFAULT_RSI.copy()

    def objective(trial):
        length = trial.suggest_int("length", 5, 30)
        os = trial.suggest_int("os", 10, 40)
        ob = trial.suggest_int("ob", 60, 90)
        if os >= ob:
            return -1e9

        rsi = _rsi(close, length).dropna()
        if rsi.empty:
            return -1e9

        hits_os = (rsi < os).mean()
        hits_ob = (rsi > ob).mean()
        coverage = hits_os + hits_ob
        if coverage == 0:
            return -1e9

        balance_penalty = abs(hits_os - hits_ob)
        return float(coverage - balance_penalty)

    vis_dir = artifacts_dir("optimized", _safe_tkr(tkr), "rsi")
    best_params = _optuna_imp_rsi.optimize_with_optuna(
        objective_func=objective,
        vis_dir=vis_dir,
        n_trials=n_trials,
        study_name=f"rsi_{tkr}",
        direction="maximize",
    )

    result = DEFAULT_RSI.copy()
    for k in ("length", "os", "ob"):
        if k in best_params:
            result[k] = int(best_params[k])

    logger.info("RSI optimized params: %s", result)
    return result


def run_optimization(
    df: pd.DataFrame,
    tkr: str,
    n_trials: int,
    mode: OptimizerMode = "macd",
) -> Dict[str, Any]:
    """
    Run optimization based on mode and return optimized parameters.

    Returns a dict with keys:
      - 'macd_params': Dict[str, int] (always present)
      - 'rsi_params': Dict[str, int] (always present)
      - 'optimizer_used': str (which optimizer was run)
    """
    close = df["Close"].astype(float).dropna()

    macd_params = DEFAULT_MACD.copy()
    rsi_params = DEFAULT_RSI.copy()

    if n_trials <= 0:
        logger.info("n_trials=%d, using default parameters", n_trials)
        return {
            "macd_params": macd_params,
            "rsi_params": rsi_params,
            "optimizer_used": "none",
        }

    if mode == "macd":
        logger.info("Running MACD optimization (%d trials)...", n_trials)
        macd_params = optimize_macd(close, tkr, n_trials)
        optimizer_used = "macd"
    elif mode == "rsi":
        logger.info("Running RSI optimization (%d trials)...", n_trials)
        rsi_params = optimize_rsi(close, tkr, n_trials)
        optimizer_used = "rsi"
    else:  # both
        logger.info("Running MACD + RSI optimization (%d trials each)...", n_trials)
        macd_params = optimize_macd(close, tkr, n_trials)
        rsi_params = optimize_rsi(close, tkr, n_trials)
        optimizer_used = "both"

    # Save optimized params to artifacts with timestamp
    from datetime import datetime as dt
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    out_dir = artifacts_dir("optimized", _safe_tkr(tkr))
    params_file = out_dir / f"{_safe_tkr(tkr)}_{timestamp}_optimized_params.json"
    params_file.write_text(
        json.dumps({
            "ticker": tkr,
            "timestamp": timestamp,
            "optimizer_used": optimizer_used,
            "n_trials": n_trials,
            "macd_params": macd_params,
            "rsi_params": rsi_params,
        }, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved optimized params to %s", params_file)

    return {
        "macd_params": macd_params,
        "rsi_params": rsi_params,
        "optimizer_used": optimizer_used,
    }


def _sanitize_config(cfg: AppConfig) -> Dict[str, Any]:
    """
    Convert the AppConfig into a plain dict suitable for serialization.
    """
    return cfg.model_dump()


def _build_overlays(df: pd.DataFrame, cfg: AppConfig) -> Dict[str, pd.Series]:
    """
    Build overlay series (e.g. SMAs) for charting.

    Currently: uses the smallest SMA window from cfg.ti_params.sma_windows,
    if provided, and overlays it on price.
    """
    overlays: Dict[str, pd.Series] = {}
    close = df["Close"]

    if cfg.ti_params.sma_windows:
        w = min(cfg.ti_params.sma_windows)
        overlays[f"SMA_{w}"] = sma(close, w)

    return overlays


def run_once(
    config_override: Optional[Dict[str, Any]] = None,
    n_trials: int = 0,
    optimizer: OptimizerMode = "macd",
) -> RunOutputDTO:
    """
    Main "analysis" pipeline.

    Steps:
    1. Build AppConfig from environment and/or overrides.
    2. Fetch OHLCV data for cfg.tkr.
    3. (Optional) Run Optuna optimization if n_trials > 0.
    4. Compute indicator snapshot using optimized or default params.
    5. Compute a moving-average crossover pattern.
    6. Generate a rule-based signal.
    7. Build a price + indicators chart.
    8. Return a RunOutputDTO with signal, score, explanation, chart (optional), and config.

    Parameters
    ----------
    config_override : dict, optional
        Override config values (tkr, start, end, etc.)
    n_trials : int
        Number of Optuna trials. 0 = use defaults, >0 = run optimizer.
    optimizer : str
        Which optimizer to run: "macd" (default), "rsi", or "both".
    """
    # 1) Resolve configuration
    cfg = AppConfig.from_env_or_defaults(**(config_override or {}))
    logger.info("Running pipeline for %s", cfg.tkr)

    # 2) Data
    df = fetch_ohlcv(cfg)

    # 3) Optimization (optional)
    opt_result = run_optimization(df, cfg.tkr, n_trials, optimizer)
    macd_params = opt_result["macd_params"]
    rsi_params = opt_result["rsi_params"]

    # Update config with optimized MACD params
    cfg = cfg.model_copy(update={
        "ti_params": cfg.ti_params.model_copy(update={
            "macd_fast": macd_params["fast"],
            "macd_slow": macd_params["slow"],
            "macd_signal": macd_params["signal"],
            "rsi_window": rsi_params["length"],
        })
    })

    # 4) Indicators (now using optimized params)
    indicator_snapshot: List[IndicatorDTO] = compute_indicator_snapshot(df, cfg)

    # 5) Pattern (MA crossover using optimized MACD fast/slow)
    fast = macd_params["fast"]
    slow = macd_params["slow"]
    label, _pattern_series = moving_average_crossover(df["Close"], fast, slow)
    patterns_snapshot = {"ma_crossover": label}

    # 6) Signal
    signal = generate_signal(df, indicator_snapshot, patterns_snapshot, None, cfg)

    # 7) Chart
    overlays = _build_overlays(df, cfg)
    chart: ChartDTO = price_with_indicators_chart(df, overlays, title=f"{cfg.tkr} Price")

    # 8) Assemble output DTO
    reasons = list(signal.reasons) if signal.reasons else []
    if n_trials > 0:
        reasons.insert(0, f"Optimized via {opt_result['optimizer_used']} ({n_trials} trials)")
    explanation = "; ".join(reasons) if reasons else "Rule-based assessment."

    out = RunOutputDTO(
        ticker=cfg.tkr,
        date=df.index[-1].to_pydatetime(),
        signal=signal.signal,  # type: ignore[assignment]
        score=signal.score,
        explanation=explanation,
        chart_base64=chart.data if cfg.store_charts_in_json else None,
        model_version=cfg.model_version,
        indicators=indicator_snapshot,
        config=_sanitize_config(cfg),
    )

    logger.info(
        "Finished for %s with signal=%s score=%.2f (optimizer=%s, trials=%d)",
        cfg.tkr,
        signal.signal,
        signal.score,
        opt_result["optimizer_used"],
        n_trials,
    )
    return out


def run_visual_workflow(
    config_override: Optional[Dict[str, Any]] = None
) -> Tuple[ChartDTO, str]:
    """
    Workflow 1: build the "ratio vs benchmark + 50D SMA" visual.

    Returns
    -------
    (chart_dto, benchmark_ticker)
    """
    cfg = AppConfig.from_env_or_defaults(**(config_override or {}))

    # warmup_days ensures we have enough history to compute ratio & SMA
    df_display, bench = fetch_with_benchmark(cfg, warmup_days=60)

    chart = relative_vs_price_chart(
        df_display,
        ticker=cfg.tkr,
        benchmark=bench,
        title=f"{cfg.tkr} vs {bench} (ratio left, price & 50D SMA right)",
    )

    return chart, bench


def run_with_report(
    config_override: Optional[Dict[str, Any]] = None,
    n_trials: int = 0,
    optimizer: OptimizerMode = "macd",
    report_out: Optional[Path] = None,
) -> Tuple[RunOutputDTO, Path]:
    """
    Run the main pipeline and generate an HTML composite report.

    Parameters
    ----------
    config_override : dict, optional
        Override config values (tkr, start, end, etc.)
    n_trials : int
        Number of Optuna trials. 0 = use defaults, >0 = run optimizer.
    optimizer : str
        Which optimizer to run: "macd" (default), "rsi", or "both".
    report_out : Path, optional
        Custom output path for HTML report.

    Returns
    -------
    (RunOutputDTO, Path)
        The run output DTO and the path to the HTML report.
    """
    from qmod.dto import SignalDTO  # local import to avoid circular

    # 1) Resolve configuration
    cfg = AppConfig.from_env_or_defaults(**(config_override or {}))
    logger.info("Running pipeline with report for %s", cfg.tkr)

    # 2) Data
    df = fetch_ohlcv(cfg)

    # 3) Optimization (optional)
    opt_result = run_optimization(df, cfg.tkr, n_trials, optimizer)
    macd_params = opt_result["macd_params"]
    rsi_params = opt_result["rsi_params"]

    # Update config with optimized MACD params
    cfg = cfg.model_copy(update={
        "ti_params": cfg.ti_params.model_copy(update={
            "macd_fast": macd_params["fast"],
            "macd_slow": macd_params["slow"],
            "macd_signal": macd_params["signal"],
            "rsi_window": rsi_params["length"],
        })
    })

    # 4) Indicators (now using optimized params)
    indicator_snapshot: List[IndicatorDTO] = compute_indicator_snapshot(df, cfg)

    # 5) Pattern (MA crossover using optimized MACD fast/slow)
    fast = macd_params["fast"]
    slow = macd_params["slow"]
    label, _pattern_series = moving_average_crossover(df["Close"], fast, slow)
    patterns_snapshot = {"ma_crossover": label}

    # 6) Signal
    signal = generate_signal(df, indicator_snapshot, patterns_snapshot, None, cfg)

    # 7) Chart
    overlays = _build_overlays(df, cfg)
    chart: ChartDTO = price_with_indicators_chart(df, overlays, title=f"{cfg.tkr} Price")

    # 8) Assemble output DTO
    reasons = list(signal.reasons) if signal.reasons else []
    if n_trials > 0:
        reasons.insert(0, f"Optimized via {opt_result['optimizer_used']} ({n_trials} trials)")
    explanation = "; ".join(reasons) if reasons else "Rule-based assessment."

    out = RunOutputDTO(
        ticker=cfg.tkr,
        date=df.index[-1].to_pydatetime(),
        signal=signal.signal,  # type: ignore[assignment]
        score=signal.score,
        explanation=explanation,
        chart_base64=chart.data if cfg.store_charts_in_json else None,
        model_version=cfg.model_version,
        indicators=indicator_snapshot,
        config=_sanitize_config(cfg),
    )

    # 9) Generate HTML report
    start_dt = df.index[0].to_pydatetime()
    end_dt = df.index[-1].to_pydatetime()

    last_signal_dto = SignalDTO(
        date=end_dt,
        ticker=cfg.tkr,
        signal=signal.signal,  # type: ignore[assignment]
        score=signal.score,
        reasons=reasons,
    )

    report_path = save_html_report(
        ticker=cfg.tkr,
        start_date=start_dt,
        end_date=end_dt,
        macd_params=macd_params,
        rsi_params=rsi_params,
        last_signal=last_signal_dto,
        df=df,
        output_path=report_out,
        layout="two_column",
        optimizer_used=opt_result["optimizer_used"],
        n_trials=n_trials,
    )

    logger.info("Generated HTML report: %s", report_path)
    logger.info(
        "Finished for %s with signal=%s score=%.2f (optimizer=%s, trials=%d)",
        cfg.tkr,
        signal.signal,
        signal.score,
        opt_result["optimizer_used"],
        n_trials,
    )

    return out, report_path


__all__ = [
    "run_once",
    "run_visual_workflow",
    "run_with_report",
    "run_optimization",
    "optimize_macd",
    "optimize_rsi",
    "DEFAULT_MACD",
    "DEFAULT_RSI",
]
