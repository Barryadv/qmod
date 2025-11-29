#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_optimizers.py
Standalone test script to verify optuna_imp.py and optuna_imp_rsi.py work independently.

Usage:
    python -m qmod.test_optimizers

This script tests both MACD and RSI optimizers with minimal dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure qmod package is importable
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd


def test_optuna_imp_macd():
    """Test the MACD optimizer (optuna_imp) independently."""
    print("\n" + "=" * 60)
    print("TEST 1: optuna_imp.py (MACD optimizer)")
    print("=" * 60)

    try:
        from qmod import optuna_imp
        print("✅ Successfully imported qmod.optuna_imp")
    except Exception as e:
        print(f"❌ Failed to import qmod.optuna_imp: {e}")
        return False

    # Check the optimize_with_optuna function exists
    if not hasattr(optuna_imp, "optimize_with_optuna"):
        print("❌ optuna_imp.optimize_with_optuna function not found")
        return False
    print("✅ Found optimize_with_optuna function")

    # Create synthetic price data for testing
    np.random.seed(42)
    n_days = 200
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    
    # Generate synthetic close prices (random walk with drift)
    returns = np.random.randn(n_days) * 0.02 + 0.0003
    close = 100 * np.exp(np.cumsum(returns))
    close_series = pd.Series(close, index=dates)

    # Define a simple MACD objective function
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    def macd_objective(trial):
        fast = trial.suggest_int("fast", 8, 15)
        slow = trial.suggest_int("slow", fast + 5, 35)
        signal = trial.suggest_int("signal", 5, 12)

        macd_line = ema(close_series, fast) - ema(close_series, slow)
        signal_line = ema(macd_line, signal)
        hist = (macd_line - signal_line).dropna()

        if hist.empty:
            return -1e9

        hret = hist.diff().dropna()
        if hret.std() == 0:
            return -1e9

        sharpe = hret.mean() / hret.std() * np.sqrt(252)
        return float(sharpe)

    # Run optimization with a small number of trials
    print("\n🔄 Running MACD optimization (5 trials)...")
    try:
        best_params = optuna_imp.optimize_with_optuna(
            objective_func=macd_objective,
            n_trials=5,
            study_name="test_macd",
            direction="maximize",
        )
        print(f"✅ MACD optimization completed!")
        print(f"   Best params: {best_params}")
        return True
    except Exception as e:
        print(f"❌ MACD optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optuna_imp_rsi():
    """Test the RSI optimizer (optuna_imp_rsi) independently."""
    print("\n" + "=" * 60)
    print("TEST 2: optuna_imp_rsi.py (RSI optimizer)")
    print("=" * 60)

    try:
        from qmod import optuna_imp_rsi
        print("✅ Successfully imported qmod.optuna_imp_rsi")
    except Exception as e:
        print(f"❌ Failed to import qmod.optuna_imp_rsi: {e}")
        return False

    # Check the optimize_with_optuna function exists
    if not hasattr(optuna_imp_rsi, "optimize_with_optuna"):
        print("❌ optuna_imp_rsi.optimize_with_optuna function not found")
        return False
    print("✅ Found optimize_with_optuna function")

    # Create synthetic price data for testing
    np.random.seed(123)
    n_days = 200
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    
    # Generate synthetic close prices
    returns = np.random.randn(n_days) * 0.025 + 0.0002
    close = 50 * np.exp(np.cumsum(returns))
    close_series = pd.Series(close, index=dates)

    # Define RSI calculation
    def calc_rsi(close: pd.Series, length: int) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def rsi_objective(trial):
        length = trial.suggest_int("length", 7, 25)
        os = trial.suggest_int("os", 15, 35)
        ob = trial.suggest_int("ob", 65, 85)

        if os >= ob:
            return -1e9

        rsi = calc_rsi(close_series, length).dropna()
        if rsi.empty:
            return -1e9

        # Simple objective: prefer RSI that oscillates well
        hits_os = (rsi < os).mean()
        hits_ob = (rsi > ob).mean()
        coverage = hits_os + hits_ob

        if coverage == 0:
            return -1e9

        balance_penalty = abs(hits_os - hits_ob)
        score = float(coverage - balance_penalty)
        return score

    # Run optimization with a small number of trials
    print("\n🔄 Running RSI optimization (5 trials)...")
    try:
        best_params = optuna_imp_rsi.optimize_with_optuna(
            objective_func=rsi_objective,
            n_trials=5,
            study_name="test_rsi",
            direction="maximize",
        )
        print(f"✅ RSI optimization completed!")
        print(f"   Best params: {best_params}")
        return True
    except Exception as e:
        print(f"❌ RSI optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("OPTUNA OPTIMIZER STANDALONE TEST")
    print("=" * 60)
    print("Testing if optuna_imp.py and optuna_imp_rsi.py work independently")
    print(f"Python: {sys.executable}")
    print(f"Working directory: {Path.cwd()}")

    results = {}
    
    # Test MACD optimizer
    results["macd"] = test_optuna_imp_macd()
    
    # Test RSI optimizer
    results["rsi"] = test_optuna_imp_rsi()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name.upper()}: {status}")

    all_passed = all(results.values())
    print("\n" + ("🎉 All tests passed!" if all_passed else "⚠️ Some tests failed"))
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

