import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qmod.paths import artifacts_dir
from qmod.data_builder import DatasetBuilder
from qmod import config as cfg_mod


# ----- Indicators -----
def calc_macd(df: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
    """
    Compute MACD using fast/slow EMA and signal EMA.
    Returns a DataFrame with columns: macd, signal, hist (indexed like df).
    Robust to df['Close'] being a Series or a 1-column DataFrame.
    """
    close = df["Close"]

    # If yfinance (or other source) gives Close as a 1-col DataFrame, squeeze it
    if isinstance(close, pd.DataFrame):
        if close.shape[1] != 1:
            raise ValueError(
                f"calc_macd expected 'Close' to be 1D, got shape {close.shape}"
            )
        close = close.iloc[:, 0]

    close = close.astype(float)
    idx = close.index

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - macd_signal

    out = pd.DataFrame(index=idx)
    out["macd"] = macd
    out["signal"] = macd_signal
    out["hist"] = hist
    return out

def detect_crossovers(macd_df: pd.DataFrame) -> pd.Series:
    """
    Return +1 on bullish MACD cross (macd crosses above signal),
           -1 on bearish cross (macd crosses below signal),
            0 otherwise.
    """
    prev_m = macd_df["macd"].shift(1)
    prev_s = macd_df["signal"].shift(1)
    cur_m = macd_df["macd"]
    cur_s = macd_df["signal"]
    bullish = ((prev_m < prev_s) & (cur_m > cur_s)).astype(int)
    bearish = -((prev_m > prev_s) & (cur_m < cur_s)).astype(int)
    return (bullish + bearish).rename("signal")

# ----- Metrics -----


def compute_metrics(returns: list[float]) -> tuple[float, float, float, float]:
    """
    Strategy trade-by-trade returns -> (cumulative, sharpe, max_drawdown, win_rate).
    Assumes each element in `returns` is a completed trade return (not daily).
    """
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Treat returns as independent periods for cumulative product
    cumulative = float(np.prod(1 + arr) - 1)

    # Build equity curve from trades to compute drawdown on a consistent 1..N index
    wealth = np.cumprod(1 + arr)
    dd = np.maximum.accumulate(wealth) - wealth
    max_dd = float(dd.max() if wealth.size else 0.0)

    # Sharpe on trade returns (scaled by sqrt(252) for rough comparability)
    std = float(arr.std(ddof=0))
    sharpe = float((arr.mean() / std) * np.sqrt(252)) if std > 0 else 0.0

    win_rate = float((arr > 0).mean())
    return cumulative, sharpe, max_dd, win_rate


# ----- Plots -----
def plot_wealth_index(
    wealth_index: pd.Series, df: pd.DataFrame, save_path: str | None = None
) -> None:
    """
    Plot strategy wealth vs. Buy & Hold (rebased).
    wealth_index should be aligned to df.index or reindexed inside.
    Robust to Close being a Series or 1-col DataFrame.
    """
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] != 1:
            raise ValueError(
                f"plot_wealth_index expected Close to be 1D, got shape {close.shape}"
            )
        close = close.iloc[:, 0]
    close = close.astype(float)

    buy_hold = (close / float(close.iloc[0])).rename("Buy & Hold")
    strategy_curve = wealth_index.reindex(df.index).ffill().bfill()

    plt.figure(figsize=(12, 5))
    plt.plot(strategy_curve.index, strategy_curve, label="Strategy")
    plt.plot(buy_hold.index, buy_hold.values, label="Buy & Hold", linestyle="--")
    plt.title("Wealth Index")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_macd_signals(
    df: pd.DataFrame,
    macd_df: pd.DataFrame,
    trades: list[dict],
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """
    Two panes: price with buy/sell markers; MACD with signal + histogram.
    """
    fig, (ax_price, ax_macd) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Price with markers
    ax_price.plot(df.index, df["Close"], label="Close", linewidth=1.2, zorder=1)

    buy_idx = [pd.to_datetime(t["date"]) for t in trades if t.get("type") == "buy"]
    sell_idx = [
        pd.to_datetime(t["date"])
        for t in trades
        if str(t.get("type", "")).startswith("sell")
    ]
    if buy_idx:
        ax_price.scatter(
            buy_idx,
            df.loc[buy_idx, "Close"],
            marker="^",
            s=70,
            label="Buy",
            zorder=3,
        )
    if sell_idx:
        ax_price.scatter(
            sell_idx,
            df.loc[sell_idx, "Close"],
            marker="v",
            s=70,
            label="Sell",
            zorder=3,
        )

    ax_price.set_title("Price with Buy/Sell Signals")
    ax_price.legend()
    ax_price.grid(True)

    # MACD panel
    ax_macd.plot(macd_df.index, macd_df["macd"], label="MACD", linewidth=1.2)
    ax_macd.plot(macd_df.index, macd_df["signal"], label="Signal", linewidth=1.2)
    ax_macd.bar(macd_df.index, macd_df["hist"], label="Histogram", alpha=0.5, width=1)
    ax_macd.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_macd.set_title("MACD, Signal, and Histogram")
    ax_macd.legend()
    ax_macd.grid(True)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.close(fig)


def plot_macd_classic(
    price_df: pd.DataFrame,
    macd_df: pd.DataFrame,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Classic two-panel MACD chart used by main.py after backtest.
    """
    fig, (ax_price, ax_macd) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax_price.plot(price_df.index, price_df["Close"], label="Close Price", color="black")
    ax_price.set_title("Price")
    ax_price.legend()
    ax_price.grid(True)

    ax_macd.plot(macd_df.index, macd_df["macd"], label="MACD", linewidth=1.2)
    ax_macd.plot(macd_df.index, macd_df["signal"], label="Signal", linewidth=1.2)
    ax_macd.bar(macd_df.index, macd_df["hist"], label="Histogram", alpha=0.5, width=1)
    ax_macd.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_macd.set_title("MACD, Signal, and Histogram")
    ax_macd.legend()
    ax_macd.grid(True)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ----- Buy & Hold Stats (utility) -----
def compute_buy_hold_stats(prices: pd.Series | pd.DataFrame) -> tuple[float, float, float, float]:
    """
    Compute buy & hold stats:
      cumulative return, annualized return, sharpe, sortino.

    Accepts either a Series or a 1-column DataFrame.
    """
    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] != 1:
            raise ValueError(f"compute_buy_hold_stats expected 1D prices, got shape {prices.shape}")
        prices = prices.iloc[:, 0]

    prices = prices.dropna()
    if prices.empty:
        return 0.0, 0.0, 0.0, 0.0

    start = float(prices.iloc[0])
    end = float(prices.iloc[-1])

    returns = prices.pct_change().dropna().values
    cumulative = float((end / start) - 1.0)
    annual_return = float((end / start) ** (252 / len(prices)) - 1.0) if len(prices) > 0 else 0.0
    sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if returns.std() > 0 else 0.0

    downside = returns[returns < 0]
    sortino = float((returns.mean() / downside.std()) * np.sqrt(252)) if downside.std() > 0 else 0.0

    return cumulative, annual_return, sharpe, sortino

# ----- Backtest -----
def run_backtest(macd_params: dict, plot: bool = False, vis_dir: str | None = None):
    """
    Runs a simple MACD crossover strategy:
      - Enter long on bullish cross, exit on bearish cross.

    Returns tuple expected by main.py:
      (cumulative, sharpe, drawdown, win_rate, trades, equity_curve,
       bh_cumulative, bh_sharpe, bh_drawdown)
    """
    # Ensure CONFIG exists (FastAPI path may not run __main__)
    if getattr(cfg_mod, "CONFIG", None) is None:
        try:
            cfg_mod.init_CONFIG()
        except Exception:
            pass
    cfg_now = cfg_mod.CONFIG or {}

    # Build dataset from the live CONFIG dict (not a copied global)
    dataset = DatasetBuilder(cfg_now)
    df = dataset.get_full_dataframe()
    if df is None or len(df) == 0:
        # safe empty return
        empty_curve = pd.Series([1.0], index=[pd.Timestamp.utcnow()], name="Wealth")
        return 0.0, 0.0, 0.0, 0.0, [], empty_curve, 0.0, 0.0, 0.0

    fast = int(macd_params["fast"])
    slow = int(macd_params["slow"])
    signal = int(macd_params["signal"])

    # Ensure we have a 1D Close series for all price access below
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] != 1:
            raise ValueError(f"run_backtest expected Close to be 1D, got shape {close.shape}")
        close = close.iloc[:, 0]
    close = close.astype(float)

    indicators = calc_macd(df, fast, slow, signal)

    crossover = detect_crossovers(indicators)

    position = 0
    entry_price = 0.0
    returns: list[float] = []
    trades: list[dict] = []

    for i in range(1, len(df)):
        sig = int(crossover.iloc[i])

        # Safe price access using the squeezed 1-D close Series
        px = float(close.iloc[i])
        date_str = str(pd.Timestamp(df.index[i]).date())

        if position == 0 and sig == 1:
            position = 1
            entry_price = px
            trades.append({"date": date_str, "type": "buy", "price": px})

        elif position == 1 and sig == -1:
            ret = (px - entry_price) / entry_price
            returns.append(float(ret))
            trades.append(
                {
                    "date": date_str,
                    "type": "sell",
                    "price": px,
                    "trade_return": float(ret),
                }
            )
            position = 0
            entry_price = 0.0

    # Close open trade at end
    if position == 1:
        last_px = float(close.iloc[-1])
        ret = (last_px - entry_price) / entry_price
        returns.append(float(ret))
        trades.append(
            {
                "date": str(pd.Timestamp(df.index[-1]).date()),
                "type": "sell_eod",
                "price": last_px,
                "trade_return": float(ret),
            }
        )


    cumulative, sharpe, drawdown, win_rate = compute_metrics(returns)

    # Build a simple equity curve indexed by sell dates (step up on each closing trade)
    wealth_steps = np.cumprod(1 + np.asarray(returns, dtype=float))
    sell_dates = [
        pd.to_datetime(t["date"])
        for t in trades
        if str(t.get("type", "")).startswith("sell")
    ]
    if len(wealth_steps) > 0 and len(sell_dates) == len(wealth_steps):
        equity_curve = pd.Series(wealth_steps, index=sell_dates, name="Wealth")
    else:
        equity_curve = pd.Series(
            [1.0], index=[pd.Timestamp(df.index[0])], name="Wealth"
        )

    # ---------- 🔧 CHANGED BLOCK: handle Close as Series ----------
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        # yfinance can return a 1-column DataFrame; squeeze to Series
        if close.shape[1] != 1:
            raise ValueError(f"run_backtest expected Close to be 1D, got shape {close.shape}")
        close = close.iloc[:, 0]
    close = close.astype(float)

    # Buy & hold stats for comparison (robust to 1-col DataFrame from yfinance)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] != 1:
            raise ValueError(f"run_backtest expected Close to be 1D, got shape {close.shape}")
        close = close.iloc[:, 0]
    close = close.astype(float)

    bh_cumulative, _bh_annual, bh_sharpe, bh_drawdown = compute_buy_hold_stats(close)

    # Build a simple buy & hold equity curve (rebased to 1.0) as a Series
    bh_equity_curve = (close / float(close.iloc[0])).rename("BuyHold_Equity")

    # ---------- end changed block ----------

    # Optional visuals & exports
    if plot:
        if vis_dir:
            os.makedirs(vis_dir, exist_ok=True)
            plot_wealth_index(
                equity_curve, df, save_path=os.path.join(vis_dir, "wealth_index.png")
            )
            plot_macd_signals(
                df,
                indicators,
                trades,
                save_path=os.path.join(vis_dir, "macd_signals.png"),
            )
        else:
            plot_wealth_index(equity_curve, df, save_path=None)
            plot_macd_signals(df, indicators, trades, save_path=None, show=True)

    # --------------------------------------------------------
    # CSV EXPORTS (Performance, Trades, Equity Curves, JSON)
    # --------------------------------------------------------
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

        ticker_val = str(cfg_now.get("ticker", "TICK"))

        # Performance Summary CSV
        perf_row = {
            "Ticker": ticker_val,
            "Strategy_Cumulative": float(cumulative),
            "Strategy_Sharpe": float(sharpe),
            "Strategy_Drawdown": float(drawdown),
            "Strategy_WinRate": float(win_rate),
            "BuyHold_Cumulative": float(bh_cumulative),
            "BuyHold_Sharpe": float(bh_sharpe),
            "BuyHold_Drawdown": float(bh_drawdown),
            "Num_Trades": int(
                len(
                    [
                        t
                        for t in trades
                        if str(t.get("type", "")).startswith("sell")
                    ]
                )
            ),
        }
        pd.DataFrame([perf_row]).to_csv(
            os.path.join(vis_dir, "performance_summary.csv"),
            index=False,
        )

        # Trades CSV
        pd.DataFrame(trades).to_csv(
            os.path.join(vis_dir, "trades.csv"),
            index=False,
        )

        # Strategy vs Buy & Hold equity curves
        eq_df = pd.DataFrame(
            {
                "Strategy_Equity": equity_curve,
                "BuyHold_Equity": bh_equity_curve.reindex(equity_curve.index).ffill(),
            }
        )
        eq_df.index.name = "Date"
        eq_df.to_csv(
            os.path.join(vis_dir, "equity_curves.csv"),
            index=True,
        )

        # summary.json
        summary = {
            "strategy": "MACD",
            "ticker": ticker_val,
            "params": {"fast": fast, "slow": slow, "signal": signal},
            "total_return": float(cumulative),
            "sharpe": float(sharpe),
            "max_drawdown": float(drawdown),
            "win_rate": float(win_rate),
            "num_trades": int(
                len(
                    [
                        t
                        for t in trades
                        if str(t.get("type", "")).startswith("sell")
                    ]
                )
            ),
            "buyhold": {
                "total_return": float(bh_cumulative),
                "sharpe": float(bh_sharpe),
                "max_drawdown": float(bh_drawdown),
            },
        }
        with open(
            os.path.join(vis_dir, "summary.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2)

    # Return the tuple main.py unpacks
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

# ----- Optuna Objective -----
def objective(trial):
    """
    Optuna objective for MACD. Suggests (fast, slow, signal),
    runs the backtest, and returns Sharpe to maximize.
    """
    fast = trial.suggest_int("fast", 4, 30)
    slow = trial.suggest_int("slow", fast + 1, 50)  # enforce fast < slow
    signal = trial.suggest_int("signal", 4, 20)

    params = {"fast": fast, "slow": slow, "signal": signal}
    cumulative, sharpe, drawdown, win_rate, *_ = run_backtest(
        params, plot=False, vis_dir=None
    )
    return float(sharpe)


if __name__ == "__main__":
    # quick self-test (writes to a ticker-aware folder under artifacts/)
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
    vis_dir = str(
        artifacts_dir(
            "vis_macd",
            f"{safe_tkr}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
    )
    os.makedirs(vis_dir, exist_ok=True)
    params = {"fast": 12, "slow": 26, "signal": 9}
    run_backtest(params, plot=True, vis_dir=vis_dir)
