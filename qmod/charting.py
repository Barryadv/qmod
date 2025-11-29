from __future__ import annotations

import base64
import io
import logging
from typing import Dict, Union

import matplotlib.pyplot as plt
import pandas as pd

from qmod.dto import ChartDTO

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def price_with_indicators_chart(
    df: pd.DataFrame,
    overlays: Dict[str, Union[pd.Series, pd.DataFrame]],
    title: str,
    show_legend: bool = True,
) -> ChartDTO:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["Close"], label="Close")
    for name, obj in overlays.items():
        if isinstance(obj, pd.Series):
            ax.plot(obj.index, obj.values, label=name)
        elif isinstance(obj, pd.DataFrame):
            for col in obj.columns:
                ax.plot(obj.index, obj[col].values, label=f"{name}:{col}")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    if show_legend:
        ax.legend(loc="best")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return ChartDTO(format="png_base64", data=b64)

def relative_vs_price_chart(
    df: pd.DataFrame,
    ticker: str,
    benchmark: str,
    title: Optional[str] = None,
) -> ChartDTO:
    """
    Dual-axis chart:
      - Left axis: RatioRebased (ticker/benchmark rebased to 100 at display start)
      - Right axis: Ticker Close and 50-day SMA

    Expects df columns: ['Close', 'SMA_50', 'RatioRebased'] and a DateTimeIndex.
    """
    if not {"Close", "SMA_50", "RatioRebased"}.issubset(df.columns):
        raise ValueError("DataFrame must include Close, SMA_50, RatioRebased columns.")

    fig, ax_right = plt.subplots(figsize=(10, 5), dpi=150)

    # Right axis: price and SMA
    ax_right.plot(df.index, df["Close"], label=f"{ticker} Close", linewidth=1.5)
    ax_right.plot(df.index, df["SMA_50"], label=f"{ticker} 50D SMA", linewidth=1.5, linestyle="--")
    ax_right.set_ylabel(f"{ticker} Price")
    ax_right.grid(True, alpha=0.3)

    # Left axis: ratio
    ax_left = ax_right.twinx()
    ax_left.plot(df.index, df["RatioRebased"], label=f"{ticker}/{benchmark} (rebased=100)", linewidth=1.5, color="green")
    ax_left.set_ylabel(f"{ticker}/{benchmark} (rebased=100)", color="green")
    ax_left.tick_params(axis='y', labelcolor="green")

    # Title & legend
    ttl = title or f"{ticker} vs {benchmark}"
    fig.suptitle(ttl, fontsize=12)

    # Build a single legend that merges handles from both axes
    handles, labels = [], []
    for ax in (ax_right, ax_left):
        h, l = ax.get_legend_handles_labels()
        handles.extend(h); labels.extend(l)
    ax_right.legend(handles, labels, loc="upper left")

    # Tight layout & encode to PNG base64
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    data64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return ChartDTO(format="png_base64", data=data64)

__all__ = ["price_with_indicators_chart"]
