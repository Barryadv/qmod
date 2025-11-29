"""
optuna_imp_combined.py
======================
Joint Optuna optimization for MACD + RSI parameters using a confluence strategy.

The objective function optimizes all 6 parameters together:
- MACD: fast, slow, signal
- RSI: length, oversold (os), overbought (ob)

The strategy uses a STATE-BASED confluence approach (symmetric):
- BUY:  MACD line > signal line (bullish) AND RSI < oversold threshold
- SELL: MACD line < signal line (bearish) AND RSI > overbought threshold

Signals are generated when ENTERING a confluence state (not on crossovers alone).
This means either indicator can "trigger first" - as long as both conditions are met.

The objective maximizes a combined performance metric (total return + win rate).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import optuna
import pandas as pd

from qmod.paths import artifacts_dir

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def _macd_hist(close: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    """MACD histogram = MACD line - Signal line."""
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    return macd_line - signal_line


def _rsi(close: pd.Series, length: int) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=length, adjust=False).mean()
    avg_loss = loss.ewm(span=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# Confluence signal generator
# ---------------------------------------------------------------------------

def generate_confluence_signals(
    close: pd.Series,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    rsi_length: int,
    rsi_os: int,
    rsi_ob: int,
) -> pd.Series:
    """
    Generate confluence signals using STATE-BASED approach (Option B):
      +1 = BUY  (MACD bullish AND RSI oversold) - signal on ENTERING this state
      -1 = SELL (MACD bearish AND RSI overbought) - signal on ENTERING this state
       0 = HOLD
    
    This is symmetric: either indicator can "trigger" as long as both conditions are met.
    Signals are generated when ENTERING the confluence state (not on every bar it's active).
    """
    macd_hist = _macd_hist(close, macd_fast, macd_slow, macd_signal)
    rsi = _rsi(close, rsi_length)
    
    # State conditions (both must be true)
    macd_bullish = macd_hist > 0  # MACD line > signal line
    macd_bearish = macd_hist < 0  # MACD line < signal line
    rsi_oversold = rsi < rsi_os
    rsi_overbought = rsi > rsi_ob
    
    # Confluence states
    buy_state = macd_bullish & rsi_oversold      # Both bullish conditions active
    sell_state = macd_bearish & rsi_overbought   # Both bearish conditions active
    
    # Signal only when ENTERING the state (state transitions from False to True)
    buy_entry = buy_state & (~buy_state.shift(1).fillna(False))
    sell_entry = sell_state & (~sell_state.shift(1).fillna(False))
    
    # Build signal series
    signals = pd.Series(0, index=close.index)
    signals[buy_entry] = 1   # BUY on entering buy confluence
    signals[sell_entry] = -1  # SELL on entering sell confluence
    
    return signals


# ---------------------------------------------------------------------------
# Backtest for objective calculation
# ---------------------------------------------------------------------------

def simple_backtest(close: pd.Series, signals: pd.Series) -> Dict[str, float]:
    """
    Simple backtest: 
    - Enter long on BUY (+1), exit on SELL (-1)
    - Calculate total return and other metrics
    
    Returns dict with: total_return, n_trades, win_rate, sharpe (if applicable)
    """
    position = 0  # 0 = flat, 1 = long
    entry_price = 0.0
    returns = []
    
    close_arr = close.values
    signal_arr = signals.values
    
    for i in range(len(close_arr)):
        sig = signal_arr[i]
        price = close_arr[i]
        
        if sig == 1 and position == 0:  # BUY
            position = 1
            entry_price = price
        elif sig == -1 and position == 1:  # SELL
            position = 0
            trade_return = (price - entry_price) / entry_price
            returns.append(trade_return)
    
    # Close any open position at end
    if position == 1:
        trade_return = (close_arr[-1] - entry_price) / entry_price
        returns.append(trade_return)
    
    if len(returns) == 0:
        return {
            "total_return": 0.0,
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
        }
    
    returns_arr = np.array(returns)
    total_return = np.prod(1 + returns_arr) - 1
    win_rate = np.sum(returns_arr > 0) / len(returns_arr)
    avg_return = np.mean(returns_arr)
    
    return {
        "total_return": float(total_return),
        "n_trades": len(returns),
        "win_rate": float(win_rate),
        "avg_return": float(avg_return),
    }


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_combined_objective(close: pd.Series) -> Callable[[optuna.Trial], float]:
    """
    Factory that returns an Optuna objective function for combined MACD+RSI optimization.
    
    The objective maximizes: total_return + 0.5 * win_rate (weighted combination)
    Penalizes too few trades to avoid degenerate solutions.
    """
    def objective(trial: optuna.Trial) -> float:
        # MACD parameters
        macd_fast = trial.suggest_int("macd_fast", 5, 20)
        macd_slow = trial.suggest_int("macd_slow", 20, 50)
        macd_signal = trial.suggest_int("macd_signal", 5, 15)
        
        # Ensure fast < slow
        if macd_fast >= macd_slow:
            return -999.0  # Invalid params
        
        # RSI parameters
        rsi_length = trial.suggest_int("rsi_length", 7, 21)
        rsi_os = trial.suggest_int("rsi_os", 20, 40)
        rsi_ob = trial.suggest_int("rsi_ob", 60, 80)
        
        # Ensure os < ob
        if rsi_os >= rsi_ob:
            return -999.0
        
        # Generate signals
        signals = generate_confluence_signals(
            close, macd_fast, macd_slow, macd_signal,
            rsi_length, rsi_os, rsi_ob
        )
        
        # Backtest
        metrics = simple_backtest(close, signals)
        
        # Penalize too few trades (want at least 3 trades for statistical relevance)
        if metrics["n_trades"] < 3:
            return -999.0 + metrics["n_trades"] * 0.1
        
        # Objective: weighted combination of return and win rate
        # This balances profitability with consistency
        score = metrics["total_return"] + 0.3 * metrics["win_rate"]
        
        return score
    
    return objective


# ---------------------------------------------------------------------------
# Main optimization function
# ---------------------------------------------------------------------------

def optimize_combined(
    close: pd.Series,
    tkr: str,
    n_trials: int = 50,
    vis_dir: Optional[Path] = None,
    study_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run joint MACD+RSI optimization using Optuna.
    
    Returns dict with:
        - macd_params: {fast, slow, signal}
        - rsi_params: {length, os, ob}
        - backtest_metrics: {total_return, n_trades, win_rate, avg_return}
    """
    if vis_dir is None:
        vis_dir = artifacts_dir("optimized", tkr.upper(), "combined")
    else:
        vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    if study_name is None:
        study_name = f"combined_{tkr}"
    
    log.info("Starting combined MACD+RSI optimization for %s (%d trials)", tkr, n_trials)
    
    # Create and run study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
    )
    
    objective = make_combined_objective(close)
    
    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Extract best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    macd_params = {
        "fast": best_params["macd_fast"],
        "slow": best_params["macd_slow"],
        "signal": best_params["macd_signal"],
    }
    
    rsi_params = {
        "length": best_params["rsi_length"],
        "os": best_params["rsi_os"],
        "ob": best_params["rsi_ob"],
    }
    
    # Run final backtest with best params for metrics
    signals = generate_confluence_signals(
        close,
        macd_params["fast"], macd_params["slow"], macd_params["signal"],
        rsi_params["length"], rsi_params["os"], rsi_params["ob"]
    )
    backtest_metrics = simple_backtest(close, signals)
    
    log.info("Combined optimization complete. Best score: %.4f", best_value)
    log.info("  MACD params: %s", macd_params)
    log.info("  RSI params: %s", rsi_params)
    log.info("  Backtest: %d trades, %.1f%% return, %.1f%% win rate",
             backtest_metrics["n_trades"],
             backtest_metrics["total_return"] * 100,
             backtest_metrics["win_rate"] * 100)
    
    # Save artifacts with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    (vis_dir / f"{timestamp}_best_params.json").write_text(
        json.dumps({
            "macd": macd_params,
            "rsi": rsi_params,
        }, indent=2),
        encoding="utf-8",
    )
    
    (vis_dir / f"{timestamp}_study_summary.json").write_text(
        json.dumps({
            "study_name": study_name,
            "direction": "maximize",
            "best_value": best_value,
            "best_params": best_params,
            "n_trials": n_trials,
            "backtest_metrics": backtest_metrics,
        }, indent=2),
        encoding="utf-8",
    )
    
    (vis_dir / f"{timestamp}_best_trial_repr.txt").write_text(
        repr(study.best_trial),
        encoding="utf-8",
    )
    
    return {
        "macd_params": macd_params,
        "rsi_params": rsi_params,
        "backtest_metrics": backtest_metrics,
        "best_score": best_value,
    }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yfinance as yf
    
    logging.basicConfig(level=logging.INFO)
    
    # Fetch test data
    tkr = "AAPL"
    df = yf.download(tkr, start="2023-01-01", end="2024-01-01", progress=False)
    close = df["Close"].squeeze()
    
    # Run optimization
    result = optimize_combined(close, tkr, n_trials=30)
    
    print("\n" + "=" * 50)
    print("COMBINED OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"MACD: fast={result['macd_params']['fast']}, "
          f"slow={result['macd_params']['slow']}, "
          f"signal={result['macd_params']['signal']}")
    print(f"RSI:  length={result['rsi_params']['length']}, "
          f"os={result['rsi_params']['os']}, "
          f"ob={result['rsi_params']['ob']}")
    print(f"Backtest: {result['backtest_metrics']['n_trades']} trades, "
          f"{result['backtest_metrics']['total_return']*100:.1f}% return, "
          f"{result['backtest_metrics']['win_rate']*100:.1f}% win rate")

