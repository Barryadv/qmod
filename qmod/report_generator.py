#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
report_generator.py
Generate human-friendly HTML reports from DTOs and artifacts.

Produces a two-column layout with:
  - Column 1: Summary (Overview, RSI reading, MACD reading, Latest Read)
  - Column 2: Charts (Price with signals, MACD, RSI)
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qmod.dto import ChartDTO, CompositeReportDTO, CompositeBulletDTO, SignalDTO, IndicatorDTO
from qmod.paths import artifacts_dir


# ---------------------------------------------------------------------------
# Chart Generation
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _macd(close: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    """Compute MACD line, signal line, and histogram."""
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    })


def _rsi(close: pd.Series, length: int) -> pd.Series:
    """Compute RSI."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def generate_price_chart(
    df: pd.DataFrame,
    ticker: str,
    macd_params: Dict[str, int],
    rsi_params: Dict[str, int],
    optimizer_used: str = "none",
) -> str:
    """Generate price chart with MACD crossover signals, return base64 PNG."""
    close = df["Close"]
    macd_df = _macd(close, macd_params["fast"], macd_params["slow"], macd_params["signal"])
    
    # Detect crossovers for signals
    bullish = (macd_df["macd"] > macd_df["signal"]) & (macd_df["macd"].shift(1) <= macd_df["signal"].shift(1))
    bearish = (macd_df["macd"] < macd_df["signal"]) & (macd_df["macd"].shift(1) >= macd_df["signal"].shift(1))
    
    fig, ax = plt.subplots(figsize=(8, 3), dpi=120)
    ax.plot(df.index, close, label="Close", color="#1f77b4", linewidth=1.2)
    
    # Plot signal markers
    ax.scatter(df.index[bullish], close[bullish], marker="^", color="green", 
               s=60, label="MACD Bullish ↑", zorder=5)
    ax.scatter(df.index[bearish], close[bearish], marker="v", color="red", 
               s=60, label="MACD Bearish ↓", zorder=5)
    
    # Build title based on optimizer used
    if optimizer_used == "macd":
        title = f"{ticker} - optimized on MACD ({macd_params['fast']}, {macd_params['slow']}, {macd_params['signal']})"
    elif optimizer_used == "rsi":
        title = f"{ticker} - optimized on RSI ({rsi_params['length']}, {rsi_params.get('os', 30)}, {rsi_params.get('ob', 70)})"
    elif optimizer_used == "both":
        title = f"{ticker} - optimized on MACD & RSI"
    else:
        title = f"{ticker} - Price (default parameters)"
    
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Price", fontsize=9)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def generate_macd_chart(
    df: pd.DataFrame,
    macd_params: Dict[str, int],
    optimizer_used: str = "none",
) -> str:
    """Generate MACD indicator chart, return base64 PNG."""
    close = df["Close"]
    macd_df = _macd(close, macd_params["fast"], macd_params["slow"], macd_params["signal"])
    
    fig, ax = plt.subplots(figsize=(8, 2.5), dpi=120)
    
    # MACD and Signal lines
    ax.plot(df.index, macd_df["macd"], label="MACD", color="#d62728", linewidth=1.2)
    ax.plot(df.index, macd_df["signal"], label="Signal", color="#1f77b4", linewidth=1.2)
    
    # Histogram as bars
    hist = macd_df["histogram"]
    colors = ["green" if v >= 0 else "#ff6b6b" for v in hist]
    ax.bar(df.index, hist, color=colors, alpha=0.6, width=1.5, label="Hist +/-")
    
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    
    # Build title with parameters
    params_str = f"({macd_params['fast']}, {macd_params['slow']}, {macd_params['signal']})"
    if optimizer_used in ("macd", "both"):
        title = f"MACD {params_str} - optimized"
    else:
        title = f"MACD {params_str}"
    
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Date", fontsize=9)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def generate_rsi_chart(
    df: pd.DataFrame,
    rsi_params: Dict[str, int],
    optimizer_used: str = "none",
) -> str:
    """Generate RSI indicator chart, return base64 PNG."""
    close = df["Close"]
    rsi = _rsi(close, rsi_params["length"])
    os_level = rsi_params.get("os", 30)
    ob_level = rsi_params.get("ob", 70)
    
    fig, ax = plt.subplots(figsize=(8, 2.5), dpi=120)
    ax.plot(df.index, rsi, label="RSI", color="#1f77b4", linewidth=1.2)
    
    # Overbought/Oversold zones
    ax.axhline(ob_level, color="#d62728", linewidth=1, linestyle="--", alpha=0.7)
    ax.axhline(os_level, color="green", linewidth=1, linestyle="--", alpha=0.7)
    ax.fill_between(df.index, ob_level, 100, alpha=0.1, color="red")
    ax.fill_between(df.index, 0, os_level, alpha=0.1, color="green")
    
    ax.set_ylim(0, 100)
    
    # Build title with parameters
    params_str = f"({rsi_params['length']}, os={os_level}, ob={ob_level})"
    if optimizer_used in ("rsi", "both"):
        title = f"RSI {params_str} - optimized"
    else:
        title = f"RSI {params_str}"
    
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Date", fontsize=9)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ---------------------------------------------------------------------------
# HTML Report Generation
# ---------------------------------------------------------------------------

def _signal_color(signal: str) -> str:
    """Return CSS color for signal."""
    return {
        "buy": "#28a745",
        "sell": "#dc3545",
        "hold": "#6c757d",
    }.get(signal.lower(), "#6c757d")


def _build_optimization_summary(
    optimizer_used: str,
    n_trials: int,
    macd_params: Dict[str, int],
    rsi_params: Dict[str, int],
) -> str:
    """Build a text summary of the optimization performed."""
    if optimizer_used == "none" or n_trials <= 0:
        return "Using default MACD &amp; RSI parameters (no optimization)."
    
    if optimizer_used == "macd":
        return (
            f"<strong>Optimized on MACD</strong> with {n_trials} trials. "
            f"Optimized parameters: fast={macd_params['fast']}, slow={macd_params['slow']}, signal={macd_params['signal']}."
        )
    elif optimizer_used == "rsi":
        return (
            f"<strong>Optimized on RSI</strong> with {n_trials} trials. "
            f"Optimized parameters: length={rsi_params['length']}, oversold={rsi_params.get('os', 30)}, overbought={rsi_params.get('ob', 70)}."
        )
    elif optimizer_used == "both":
        return (
            f"<strong>Optimized on both MACD &amp; RSI</strong> with {n_trials} trials each. "
            f"MACD: fast={macd_params['fast']}, slow={macd_params['slow']}, signal={macd_params['signal']}. "
            f"RSI: length={rsi_params['length']}, oversold={rsi_params.get('os', 30)}, overbought={rsi_params.get('ob', 70)}."
        )
    else:
        return "Using current MACD &amp; RSI parameters."


def generate_html_report(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    macd_params: Dict[str, int],
    rsi_params: Dict[str, int],
    last_signal: Optional[SignalDTO],
    df: pd.DataFrame,
    layout: Literal["two_column", "stacked"] = "two_column",
    optimizer_used: str = "none",
    n_trials: int = 0,
) -> str:
    """
    Generate a complete HTML report with two-column layout.
    
    Returns HTML string.
    """
    # Generate charts
    price_chart_b64 = generate_price_chart(df, ticker, macd_params, rsi_params, optimizer_used)
    macd_chart_b64 = generate_macd_chart(df, macd_params, optimizer_used)
    rsi_chart_b64 = generate_rsi_chart(df, rsi_params, optimizer_used)
    
    # Compute latest indicator values
    close = df["Close"]
    macd_df = _macd(close, macd_params["fast"], macd_params["slow"], macd_params["signal"])
    rsi = _rsi(close, rsi_params["length"])
    
    latest_rsi = rsi.iloc[-1]
    latest_macd = macd_df["macd"].iloc[-1]
    latest_signal_val = macd_df["signal"].iloc[-1]
    latest_date = df.index[-1].strftime("%Y-%m-%d")
    
    # Signal info
    signal_text = "N/A"
    signal_score = 0.0
    signal_color = "#6c757d"
    if last_signal:
        signal_text = last_signal.signal.upper()
        signal_score = last_signal.score
        signal_color = _signal_color(last_signal.signal)
    
    # Build HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{ticker} – Composite Report</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f8f9fa;
            color: #333;
            line-height: 1.6;
            padding: 24px;
        }}
        .header {{
            margin-bottom: 24px;
        }}
        .header h1 {{
            font-size: 28px;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 8px;
        }}
        .header-meta {{
            font-size: 14px;
            color: #666;
        }}
        .header-meta strong {{
            color: #333;
        }}
        .signal-badge {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 4px;
            font-weight: 600;
            color: white;
        }}
        .container {{
            display: {"grid" if layout == "two_column" else "flex"};
            {"grid-template-columns: 1fr 1.5fr;" if layout == "two_column" else "flex-direction: column;"}
            gap: 24px;
        }}
        .column {{
            background: white;
            border-radius: 8px;
            padding: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .column h2 {{
            font-size: 20px;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 2px solid #e9ecef;
        }}
        .section {{
            margin-bottom: 20px;
        }}
        .section h3 {{
            font-size: 15px;
            font-weight: 600;
            color: #495057;
            margin-bottom: 8px;
        }}
        .section p {{
            font-size: 14px;
            color: #555;
        }}
        .chart-container {{
            margin-bottom: 16px;
        }}
        .chart-container img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
            border: 1px solid #e9ecef;
        }}
        .params {{
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 13px;
            background: #f1f3f4;
            padding: 2px 6px;
            border-radius: 3px;
        }}
        .latest-read {{
            background: #f8f9fa;
            padding: 12px;
            border-radius: 6px;
            border-left: 4px solid {signal_color};
        }}
        .latest-read p {{
            margin-bottom: 4px;
        }}
        .interpretation {{
            font-weight: 600;
            color: {signal_color};
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{ticker} – Composite Report</h1>
        <div class="header-meta">
            <strong>Display:</strong> {start_date.strftime("%Y-%m-%d")} → {end_date.strftime("%Y-%m-%d")} &nbsp;&nbsp;
            <strong>Layout:</strong> {layout}<br>
            <strong>MACD:</strong> <span class="params">{macd_params}</span> &nbsp;&nbsp;
            <strong>RSI:</strong> <span class="params">{rsi_params}</span> &nbsp;&nbsp;
            <strong>Last Signal:</strong> <span class="signal-badge" style="background:{signal_color}">{signal_text}</span> (score={signal_score:.1f})
        </div>
    </div>

    <div class="container">
        <div class="column">
            <h2>Summary</h2>
            
            <div class="section">
                <h3>Overview</h3>
                <p>Technical picture for {ticker.lower()} from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}. {_build_optimization_summary(optimizer_used, n_trials, macd_params, rsi_params)}</p>
            </div>
            
            <div class="section">
                <h3>Reading RSI</h3>
                <p>RSI measures momentum on a 0–100 scale. Commonly: oversold &lt; {rsi_params.get("os", 30)} (buy zone), overbought &gt; {rsi_params.get("ob", 70)} (sell zone). Current parameters: length={rsi_params["length"]}, overbought={rsi_params.get("ob", 70)}, oversold={rsi_params.get("os", 30)}.</p>
            </div>
            
            <div class="section">
                <h3>Reading MACD</h3>
                <p>MACD tracks trend momentum via EMA differences. When MACD is above its signal line, momentum is strengthening; below it, weakening. Current parameters: fast={macd_params["fast"]}, slow={macd_params["slow"]}, signal={macd_params["signal"]}.</p>
            </div>
            
            <div class="section">
                <h3>Latest Read</h3>
                <div class="latest-read">
                    <p>As of {latest_date}: RSI={latest_rsi:.1f}; MACD={latest_macd:.4f} vs Signal={latest_signal_val:.4f}.</p>
                    <p class="interpretation">Composite interpretation: {signal_text}.</p>
                </div>
            </div>
        </div>

        <div class="column">
            <h2>Chart</h2>
            
            <div class="chart-container">
                <img src="data:image/png;base64,{price_chart_b64}" alt="{ticker} Price Chart">
            </div>
            
            <div class="chart-container">
                <img src="data:image/png;base64,{macd_chart_b64}" alt="MACD Chart">
            </div>
            
            <div class="chart-container">
                <img src="data:image/png;base64,{rsi_chart_b64}" alt="RSI Chart">
            </div>
        </div>
    </div>
</body>
</html>'''
    
    return html


def save_html_report(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    macd_params: Dict[str, int],
    rsi_params: Dict[str, int],
    last_signal: Optional[SignalDTO],
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    layout: Literal["two_column", "stacked"] = "two_column",
    optimizer_used: str = "none",
    n_trials: int = 0,
) -> Path:
    """
    Generate and save an HTML report.
    
    If output_path is not provided, saves to artifacts/reports/<ticker>/<ticker>_report.html
    
    Returns the path to the saved file.
    """
    html = generate_html_report(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        macd_params=macd_params,
        rsi_params=rsi_params,
        last_signal=last_signal,
        df=df,
        layout=layout,
        optimizer_used=optimizer_used,
        n_trials=n_trials,
    )
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = artifacts_dir("reports", ticker.upper())
        output_path = out_dir / f"{ticker.upper()}_{timestamp}_report.html"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_path.write_text(html, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Convenience function for CLI integration
# ---------------------------------------------------------------------------

def generate_report_from_pipeline(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    macd_params: Dict[str, int],
    rsi_params: Dict[str, int],
    signal: str,
    score: float,
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Convenience function to generate report from pipeline output.
    """
    last_signal = SignalDTO(
        date=end_date,
        ticker=ticker,
        signal=signal,
        score=score,
        reasons=[],
    )
    
    return save_html_report(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        macd_params=macd_params,
        rsi_params=rsi_params,
        last_signal=last_signal,
        df=df,
        output_path=output_path,
    )


__all__ = [
    "generate_html_report",
    "save_html_report",
    "generate_report_from_pipeline",
    "generate_price_chart",
    "generate_macd_chart",
    "generate_rsi_chart",
]

