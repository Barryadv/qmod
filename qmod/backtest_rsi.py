# backtest_rsi.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qmod.paths import artifacts_dir
from qmod.data_builder import DatasetBuilder
from qmod import config as cfg_mod


# ---------- Helpers ----------

def calc_rsi(df: pd.DataFrame, period: int) -> pd.Series:
    """Classic Wilder RSI on Close."""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.bfill().ffill().fillna(50.0)


def _compute_metrics(returns: List[float]) -> Tuple[float, float, float, float]:
    """
    Returns: cumulative, sharpe, max_drawdown, win_rate
    """
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    cumulative = float(np.prod(1 + arr) - 1)
    std = float(arr.std(ddof=0))
    sharpe = float((arr.mean() / std) * np.sqrt(252)) if std > 0 else 0.0

    wealth = np.cumprod(1 + arr)
    drawdown = float(
        (np.maximum.accumulate(wealth) - wealth).max() if wealth.size else 0.0
    )
    win_rate = float((arr > 0).mean() if arr.size else 0.0)
    return cumulative, sharpe, drawdown, win_rate


def plot_rsi_classic(
    df: pd.DataFrame,
    rsi: pd.Series,
    lower: float,
    upper: float,
    save_path: Path | str | None = None,
    show: bool = True,
) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    # Price
    ax1.plot(df.index, df["Close"], label="Close", linewidth=1.2)
    ax1.set_title("Price")
    ax1.grid(True)
    ax1.legend()

    # RSI
    ax2.plot(df.index, rsi, label="RSI", linewidth=1.2)
    ax2.axhline(lower, linestyle="--", linewidth=1.0, label=f"Lower {lower}")
    ax2.axhline(upper, linestyle="--", linewidth=1.0, label=f"Upper {upper}")
    ax2.set_ylim(0, 100)
    ax2.set_title("RSI")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p)
        plt.close(fig)
    elif show:
        plt.show()


def _plot_rsi_signals(
    df: pd.DataFrame,
    rsi: pd.Series,
    trades: list[dict],
    lower: float,
    upper: float,
    save_path: Path | str | None,
) -> None:
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Price with markers
    axs[0].plot(df.index, df["Close"], label="Close Price")
    for t in trades:
        axs[0].plot(
            t["index"],
            t["price"],
            marker="^" if t["type"] == "buy" else "v",
        )
    axs[0].set_title("Price and Trades")
    axs[0].legend()
    axs[0].grid(True)

    # RSI panel
    axs[1].plot(df.index, rsi, label="RSI", linewidth=1.2)
    axs[1].axhline(lower, linestyle="--", linewidth=1.0, label=f"Lower {lower}")
    axs[1].axhline(upper, linestyle="--", linewidth=1.0, label=f"Upper {upper}")
    axs[1].set_ylim(0, 100)
    axs[1].set_title("RSI")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p)
        plt.close(fig)
    else:
        plt.show()


def _plot_wealth_index(
    wealth_index: pd.Series,
    df: pd.DataFrame,
    save_path: Path | str | None,
) -> None:
    bh = (df["Close"] / df["Close"].iloc[0]).rename("Buy & Hold")
    curve = wealth_index.reindex(df.index).ffill().bfill()

    plt.figure(figsize=(12, 5))
    plt.plot(curve.index, curve, label="Strategy")
    plt.plot(bh.index, bh, label="Buy & Hold", linestyle="--")
    plt.title("Strategy vs. Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p)
        plt.close()
    else:
        plt.show()


# ----- Backtest -----


def run_backtest(
    rsi_params: Dict[str, float | int],
    plot: bool = False,
    vis_dir: str | Path | None = None,
):
    """
    rsi_params: {"period": int, "lower": float, "upper": float}
    Returns (9-tuple, same order as MACD):
      cumulative, sharpe, max_drawdown, win_rate,
      trades, equity_curve,
      bh_cumulative, bh_sharpe, bh_drawdown
    """
    # Ensure CONFIG exists (FastAPI path may not run __main__)
    if getattr(cfg_mod, "CONFIG", None) is None:
        try:
            cfg_mod.init_CONFIG()
        except Exception:
            pass
    cfg_now = cfg_mod.CONFIG or {}

    # Load data via DatasetBuilder
    dataset = DatasetBuilder(cfg_now)
    df = dataset.get_full_dataframe()
    if df is None or len(df) == 0:
        # Safe empty return
        empty_curve = pd.Series([1.0], index=[pd.Timestamp.utcnow()], name="Wealth")
        return 0.0, 0.0, 0.0, 0.0, [], empty_curve, 0.0, 0.0, 0.0

    period = int(rsi_params["period"])
    lower = float(rsi_params["lower"])
    upper = float(rsi_params["upper"])

    rsi = calc_rsi(df, period=period)

    # Generate trades: buy when RSI crosses up through lower; sell when RSI crosses down through upper
    pos = 0
    entry_px = 0.0
    returns: List[float] = []
    trades: List[dict] = []

    rsi_prev = rsi.shift(1)
    buy_signals = (rsi_prev <= lower) & (rsi > lower)
    sell_signals = (rsi_prev >= upper) & (rsi < upper)

    for i in range(1, len(df)):
        price = float(df["Close"].iloc[i])
        idx = df.index[i]

        if pos == 0 and bool(buy_signals.iloc[i]):
            pos = 1
            entry_px = price
            trades.append({"index": idx, "price": price, "type": "buy"})

        elif pos == 1 and bool(sell_signals.iloc[i]):
            rtn = (price - entry_px) / entry_px
            returns.append(float(rtn))
            trades.append({"index": idx, "price": price, "type": "sell", "trade_return": float(rtn)})
            pos = 0

    # Liquidate at end if in position
    if pos == 1:
        price = float(df["Close"].iloc[-1])
        rtn = (price - entry_px) / entry_px
        returns.append(float(rtn))
        trades.append(
            {
                "index": df.index[-1],
                "price": price,
                "type": "sell_eod",
                "trade_return": float(rtn),
            }
        )

    # Strategy metrics
    cumulative, sharpe, drawdown, win_rate = _compute_metrics(returns)

    # Build equity curve aligned to sell (realization) dates
    sell_dates = [t["index"] for t in trades if t["type"].startswith("sell")]
    min_len = min(len(returns), len(sell_dates))
    if min_len > 0:
        equity_curve = pd.Series(
            np.cumprod(1 + np.asarray(returns[:min_len], dtype=float)),
            index=pd.Index(sell_dates[:min_len], name="Date"),
            name="Wealth",
        )
    else:
        equity_curve = pd.Series(
            [1.0], index=[pd.Timestamp(df.index[0])], name="Wealth"
        )

    # Buy & Hold metrics (daily returns)
    bh_returns = df["Close"].pct_change().dropna().values
    bh_cumulative, bh_sharpe, bh_drawdown, _ = _compute_metrics(bh_returns)

    # Buy & Hold equity curve rebased to 1.0 for CSV
    bh_equity_curve = (df["Close"] / df["Close"].iloc[0]).rename("BuyHold_Equity")

    # ----- Output artifacts (path-safe) -----
    if plot and vis_dir is None:
        safe = (
            str(cfg_now.get("ticker", "TICK"))
            .replace(":", "_")
            .replace("-", "_")
            .replace("/", "_")
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vis_path = artifacts_dir("vis_rsi", f"{safe}_{timestamp}")
    elif vis_dir is not None:
        vis_path = Path(vis_dir)
        vis_path.mkdir(parents=True, exist_ok=True)
    else:
        vis_path = None

    if vis_path:
        ticker_val = str(cfg_now.get("ticker", "TICK"))

        # Performance CSV
        pd.DataFrame(
            [
                {
                    "Ticker": ticker_val,
                    "Strategy_Cumulative": float(cumulative),
                    "Strategy_Sharpe": float(sharpe),
                    "Strategy_Drawdown": float(drawdown),
                    "Strategy_WinRate": float(win_rate),
                    "BuyHold_Cumulative": float(bh_cumulative),
                    "BuyHold_Sharpe": float(bh_sharpe),
                    "BuyHold_Drawdown": float(bh_drawdown),
                    "RSI_Period": int(period),
                    "RSI_Lower": float(lower),
                    "RSI_Upper": float(upper),
                }
            ]
        ).to_csv(vis_path / "performance_summary.csv", index=False)

        # Trades CSV
        pd.DataFrame(trades).to_csv(vis_path / "trades.csv", index=False)

        # Strategy vs Buy & Hold equity curves CSV
        eq_df = pd.DataFrame(
            {
                "Strategy_Equity": equity_curve,
                "BuyHold_Equity": bh_equity_curve.reindex(equity_curve.index).ffill(),
            }
        )
        eq_df.index.name = "Date"
        eq_df.to_csv(vis_path / "equity_curves.csv", index=True)

        # summary.json (parity with MACD style)
        summary = {
            "strategy": "RSI",
            "ticker": ticker_val,
            "params": {
                "period": int(period),
                "lower": float(lower),
                "upper": float(upper),
            },
            "total_return": float(cumulative),
            "sharpe": float(sharpe),
            "max_drawdown": float(drawdown),
            "win_rate": float(win_rate),
            "num_trades": int(len(trades)),
            "buyhold": {
                "total_return": float(bh_cumulative),
                "sharpe": float(bh_sharpe),
                "max_drawdown": float(bh_drawdown),
            },
        }
        (vis_path / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        # Plots
        plot_rsi_classic(
            df,
            rsi,
            lower,
            upper,
            save_path=vis_path / "rsi_classic.png",
            show=False,
        )
        _plot_rsi_signals(
            df,
            rsi,
            trades,
            lower,
            upper,
            save_path=vis_path / "rsi_signals.png",
        )
        _plot_wealth_index(
            equity_curve,
            df,
            save_path=vis_path / "wealth_index.png",
        )

    # 9-tuple contract (unchanged)
    return (
        float(cumulative),
        float(sharpe),
        float(drawdown),
        float(win_rate),
        trades,
        equity_curve,
        float(bh_cumulative),
        float(bh_sharpe),
        float(bh_drawdown),
    )


def objective(trial):
    """
    Optuna objective for RSI. Suggests (period, lower, upper),
    runs the backtest, and returns Sharpe to maximize.
    """
    period = trial.suggest_int("period", 5, 40)
    lower = trial.suggest_float("lower", 10.0, 40.0)
    upper = trial.suggest_float("upper", 60.0, 90.0)

    params = {"period": period, "lower": lower, "upper": upper}
    cumulative, sharpe, drawdown, win_rate, *_ = run_backtest(
        params, plot=False, vis_dir=None
    )
    return float(sharpe)


if __name__ == "__main__":
    # Quick self-test (writes under artifacts/vis_rsi/)
    if getattr(cfg_mod, "CONFIG", None) is None:
        try:
            cfg_mod.init_CONFIG()
        except Exception:
            pass
    cfg_now = cfg_mod.CONFIG or {}
    safe_tkr = (
        str(cfg_now.get("ticker", "TICK"))
        .replace(":", "_")
        .replace("-", "_")
        .replace("/", "_")
    )
    vis_dir = artifacts_dir(
        "vis_rsi",
        f"{safe_tkr}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    params = {"period": 14, "lower": 30.0, "upper": 70.0}

    (
        cumulative,
        sharpe,
        drawdown,
        win_rate,
        trades,
        equity_curve,
        bh_cumulative,
        bh_sharpe,
        bh_drawdown,
    ) = run_backtest(params, plot=True, vis_dir=vis_dir)

    print("RSI self-test complete:")
    print(
        f"  cumulative={cumulative:.4f}, "
        f"sharpe={sharpe:.3f}, "
        f"drawdown={drawdown:.3f}, "
        f"win_rate={win_rate:.2%}"
    )
    print(f"  artifacts written to: {vis_dir}")
