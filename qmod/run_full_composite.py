#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_full_composite.py
Builds a composite JSON (params, bullets, last_signal, optional chart)
and optionally renders report.html + chart.png into artifacts/composite/<TICKER>/.

Usage (examples):
  python -m qmod.run_full_composite --tkr F --start 2024-01-01 --end 2024-03-01 --n-trials 1 --pretty ^
    --out-dir ".\artifacts\composite\F" --out-html ".\artifacts\composite\F\report.html" --out-png ".\artifacts\composite\F\chart.png"

Legacy style (write into runs/ and keep stdout printable):
  python -m qmod.run_full_composite --tkr F --start 2024-01-01 --end 2024-03-01 --n-trials 1 --pretty ^
    --out-dir ".\runs\F" --out-json ".\runs\F\F_2024-01-01_2024-03-01.json"
"""

from __future__ import annotations

# --- Headless Matplotlib guard (must be before any matplotlib import) ---
import os as _os
_os.environ["MPLBACKEND"] = "Agg"
import matplotlib as _mpl
_mpl.use("Agg", force=True)

import argparse
import asyncio
import base64
import dataclasses
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from qmod.data_importer import fetch_ohlcv
from qmod.paths import artifacts_dir

# --- Optional: colored logs off; simple logging ---
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# --- Defaults used when n_trials == 0 or optimizer unavailable ---
DEFAULT_MACD = {"fast": 12, "slow": 26, "signal": 9}
DEFAULT_RSI = {"length": 14, "os": 30, "ob": 70}

# --- Import project config if present ---
try:
    import config as cfg_mod  # your project CONFIG module
except Exception:  # noqa: BLE001
    cfg_mod = None

# ------------------------------------------------------------
# Optuna helper modules (MACD & RSI)
# ------------------------------------------------------------
_optuna_imp = None
_optuna_imp_rsi = None

# 1) Preferred: import from package (python -m qmod.run_full_composite)
try:
    from qmod import optuna_imp as _optuna_imp  # MACD optuna helper
    from qmod import optuna_imp_rsi as _optuna_imp_rsi  # RSI optuna helper
    log.info("[run_full_composite] Loaded optuna modules from package (qmod.optuna_imp, qmod.optuna_imp_rsi)")
except Exception as e:
    log.warning("[run_full_composite] Package import error for optuna modules: %r", e)
    _optuna_imp = None
    _optuna_imp_rsi = None

# 2) Fallback: local imports when running as `python qmod\\run_full_composite.py`
if _optuna_imp is None or _optuna_imp_rsi is None:
    try:
        import optuna_imp as _optuna_imp  # type: ignore[no-redef]
        import optuna_imp_rsi as _optuna_imp_rsi  # type: ignore[no-redef]
        log.info("[run_full_composite] Fallback: loaded optuna modules via local imports.")
    except Exception as e:
        log.warning("[run_full_composite] Local import error for optuna modules: %r", e)
        _optuna_imp = None
        _optuna_imp_rsi = None


def print_kv(msg: str, val: Any | None = None) -> None:
    if val is None:
        log.info(msg)
    else:
        log.info("%s: %s", msg, val)


def _safe_tkr(tkr: str) -> str:
    return tkr.replace(":", "_").replace("-", "_").replace("/", "_").strip()


# --- DTOs for final composite payload ---
@dataclasses.dataclass
class ChartDTO:
    format: str  # e.g., "png_base64"
    data: str    # base64-encoded PNG


@dataclasses.dataclass
class LastSignalDTO:
    date: datetime
    ticker: str
    signal: str
    score: float
    reasons: List[str]


@dataclasses.dataclass
class BulletDTO:
    title: str
    body: str


@dataclasses.dataclass
class CompositeReportDTO:
    ticker: str
    start_date: datetime
    end_date: datetime
    macd_params: Dict[str, int]
    rsi_params: Dict[str, int]
    bullets: List[BulletDTO]
    last_signal: LastSignalDTO
    chart: Optional[ChartDTO] = None

    def to_json(self, indent: Optional[int] = None) -> str:
        def _default(o):
            if isinstance(o, (datetime, date)):
                return o.isoformat()
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            raise TypeError(f"Not JSON serializable: {type(o)}")

        return json.dumps(self, default=_default, indent=indent)


# --- Argument parsing ---
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build composite JSON (and optional HTML/PNG report).")
    p.add_argument("--tkr", "--ticker", dest="tkr", required=True, help="Ticker symbol, e.g., F")
    p.add_argument("--start", "--start_date", dest="start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", "--end_date", dest="end", required=True, help="YYYY-MM-DD")
    p.add_argument(
        "--n-trials",
        dest="n_trials",
        type=int,
        default=0,
        help="Optuna trials (0 = use defaults)",
    )
    p.add_argument(
        "--layout",
        default="two_column",
        choices=["two_column", "stacked"],
        help="Report layout hint",
    )
    p.add_argument("--include-chart", action="store_true", help="Embed chart (base64) in JSON/report")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON")

    # Explicit output controls
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="(legacy) write JSON to this exact path",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Base dir for JSON/report (default: artifacts/composite/<TICKER>)",
    )
    p.add_argument("--out-json", type=str, default=None, help="Exact path for JSON (overrides out-dir)")
    p.add_argument("--out-html", type=str, default=None, help="Path to write HTML report")
    p.add_argument("--out-png", type=str, default=None, help="Path to write decoded chart.png")

    return p.parse_args()


# --- Date parsing ---
def parse_date(s: str) -> datetime:
    # Accept common formats; normalize to datetime
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    try:
        s2 = s.replace("Z", "").replace("z", "")
        return datetime.fromisoformat(s2)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Unrecognized date format: {s}") from e


# --- Chart building stub (replace with your actual chart rendering if needed) ---
def render_chart_b64(ticker: str, start: datetime, end: datetime, layout: str) -> str:
    """
    Return base64-encoded PNG data. Here we create a tiny placeholder PNG to keep this script self-contained.
    Replace with your real renderer if you already have one.
    """
    import io
    import matplotlib.pyplot as plt

    x = np.linspace(0, 6.283, 200)
    y = np.sin(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.set_title(f"{ticker} • {start.date()} → {end.date()} ({layout})")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=144, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ------------------------------------------------------------
# Price/helper functions for MACD / RSI objectives
# ------------------------------------------------------------
def _load_price_frame(tkr: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Minimal loader for OHLCV used by MACD/RSI objectives.
    Reuses your qmod.data_importer.fetch_ohlcv via AppConfig.
    """
    from qmod.config import AppConfig

    cfg = AppConfig.from_env_or_defaults(tkr=tkr, start=start.date(), end=end.date())
    df = fetch_ohlcv(cfg)
    if "Close" not in df.columns:
        raise ValueError("Downloaded frame has no 'Close' column.")
    return df


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _macd_hist(close: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    """
    Basic MACD histogram: EMA(fast) - EMA(slow) minus signal EMA.
    """
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return hist

def _rsi(close: pd.Series, length: int) -> pd.Series:
    """
    Classic Wilder-style RSI.
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- MACD/RSI optimization wrappers (direct Optuna) ---
def _optimize_macd(tkr: str, start: datetime, end: datetime, n_trials: int) -> Dict[str, int]:
    """
    Run Optuna for MACD parameters (fast/slow/signal) using optuna_imp directly.
    Falls back to DEFAULT_MACD if Optuna is unavailable or fails.
    """
    if n_trials is None or n_trials <= 0:
        return DEFAULT_MACD

    if _optuna_imp is None:
        log.warning("MACD optuna module not available; using defaults.")
        return DEFAULT_MACD

    try:
        df = _load_price_frame(tkr, start, end)
        close = df["Close"].astype(float).dropna()

        def objective(trial):
            # Search space – tweak as you like
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

        vis_dir = artifacts_dir("vis_macd", "composite", _safe_tkr(tkr))
        best_params = _optuna_imp.optimize_with_optuna(
            objective_func=objective,
            vis_dir=vis_dir,
            n_trials=int(n_trials),
            study_name=f"macd_{tkr}_{start:%Y%m%d}_{end:%Y%m%d}",
            direction="maximize",
        )

        params = DEFAULT_MACD.copy()
        for k in ("fast", "slow", "signal"):
            if k in best_params:
                params[k] = int(best_params[k])

        log.info("MACD Optuna best params: %s", params)
        return params

    except Exception as e:  # noqa: BLE001
        log.exception("MACD optimization failed, falling back to defaults: %s", e)
        return DEFAULT_MACD


def _optimize_rsi(tkr: str, start: datetime, end: datetime, n_trials: int) -> Dict[str, int]:
    """
    Run Optuna for RSI parameters (length/os/ob) using optuna_imp_rsi directly.
    Falls back to DEFAULT_RSI if Optuna is unavailable or fails.
    """
    if n_trials is None or n_trials <= 0:
        return DEFAULT_RSI

    if _optuna_imp_rsi is None:
        log.warning("RSI optuna module not available; using defaults.")
        return DEFAULT_RSI

    try:
        df = _load_price_frame(tkr, start, end)
        close = df["Close"].astype(float).dropna()

        def objective(trial):
            length = trial.suggest_int("length", 5, 30)
            os = trial.suggest_int("os", 10, 40)
            ob = trial.suggest_int("ob", 60, 90)
            if os >= ob:
                return -1e9

            rsi = _rsi(close, length).dropna()
            if rsi.empty:
                return -1e9

            # Simple objective: prefer RSI that oscillates and spends
            # some time in both OS and OB zones.
            hits_os = (rsi < os).mean()
            hits_ob = (rsi > ob).mean()
            coverage = hits_os + hits_ob
            if coverage == 0:
                return -1e9

            # Penalize extreme imbalance (all OS or all OB)
            balance_penalty = abs(hits_os - hits_ob)
            score = float(coverage - balance_penalty)
            return score

        vis_dir = artifacts_dir("vis_rsi", "composite", _safe_tkr(tkr))
        best_params = _optuna_imp_rsi.optimize_with_optuna(
            objective_func=objective,
            vis_dir=vis_dir,
            n_trials=int(n_trials),
            study_name=f"rsi_{tkr}_{start:%Y%m%d}_{end:%Y%m%d}",
            direction="maximize",
        )

        params = DEFAULT_RSI.copy()
        for k in ("length", "os", "ob"):
            if k in best_params:
                params[k] = int(best_params[k])

        log.info("RSI Optuna best params: %s", params)
        return params

    except Exception as e:  # noqa: BLE001
        log.exception("RSI optimization failed, falling back to defaults: %s", e)
        return DEFAULT_RSI

async def run_optimizers_async(
    tkr: str,
    start: datetime,
    end: datetime,
    n_trials: int,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    loop = asyncio.get_running_loop()
    macd_future = loop.run_in_executor(None, _optimize_macd, tkr, start, end, n_trials)
    rsi_future = loop.run_in_executor(None, _optimize_rsi, tkr, start, end, n_trials)
    macd_params, rsi_params = await asyncio.gather(macd_future, rsi_future)
    return macd_params, rsi_params


# --- Simple “last signal” stub (replace with your project’s signal engine if available) ---
def compute_last_signal(
    tkr: str,
    start: datetime,
    end: datetime,
    macd_params: Dict[str, int],
    rsi_params: Dict[str, int],
) -> LastSignalDTO:
    # Placeholder logic; wire to your real signal engine if you have it.
    sig = "buy" if (macd_params["fast"] < macd_params["slow"]) else "hold"
    score = 0.5 if sig == "hold" else 0.8
    reasons = [
        f"MACD fast={macd_params['fast']} slow={macd_params['slow']}",
        f"RSI len={rsi_params['length']}",
    ]
    return LastSignalDTO(date=end, ticker=tkr, signal=sig, score=score, reasons=reasons)


# --- HTML rendering (simple, self-contained) ---
def _render_simple_html(report: CompositeReportDTO) -> str:
    parts: List[str] = []
    parts.append(f"<h1>{report.ticker} — {report.start_date.date()} to {report.end_date.date()}</h1>")
    parts.append("<h3>Parameters</h3><ul>")
    parts.append(
        f"<li>MACD: fast={report.macd_params['fast']}, "
        f"slow={report.macd_params['slow']}, signal={report.macd_params['signal']}</li>"
    )
    parts.append(
        f"<li>RSI: length={report.rsi_params['length']}, "
        f"os={report.rsi_params['os']}, ob={report.rsi_params['ob']}</li>"
    )
    parts.append("</ul>")
    parts.append("<h3>Last signal</h3>")
    parts.append(
        f"<p>{report.last_signal.date.date()} — "
        f"<b>{report.last_signal.signal.upper()}</b> (score {report.last_signal.score})<br>"
    )
    parts.append("Reasons: " + ", ".join(report.last_signal.reasons) + "</p>")
    if report.bullets:
        parts.append("<h3>Notes</h3><ul>")
        for b in report.bullets:
            parts.append(f"<li><b>{b.title}:</b> {b.body}</li>")
        parts.append("</ul>")
    if report.chart and report.chart.format == "png_base64":
        parts.append(
            f'<img alt="chart" style="max-width:100%;" '
            f'src="data:image/png;base64,{report.chart.data}">'
        )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Composite Report</title>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;line-height:1.45}</style>"
        "</head><body>"
        + "\n".join(parts)
        + "</body></html>"
    )
    return html


def _write_png_from_b64(p: Path, b64: str) -> None:
    p.write_bytes(base64.b64decode(b64))


def _default_composite_dir(tkr: str) -> Path:
    return artifacts_dir("composite", _safe_tkr(tkr))


def _resolve_out_paths(
    args: argparse.Namespace,
    chart_b64: Optional[str],
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    base_dir = Path(args.out_dir) if args.out_dir else _default_composite_dir(args.tkr)
    base_dir.mkdir(parents=True, exist_ok=True)

    # JSON path
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
    elif args.out:
        out_json = Path(args.out)
        out_json.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_json = base_dir / f"{args.tkr}_{args.start}_{args.end}.json"

    # HTML/PNG paths (optional)
    out_html = Path(args.out_html) if args.out_html else None
    if out_html:
        out_html.parent.mkdir(parents=True, exist_ok=True)

    out_png = Path(args.out_png) if args.out_png else None
    if out_png:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        if not chart_b64:
            log.warning("--out-png requested but no chart base64 present; will write nothing.")

    return out_json, out_html, out_png


def main() -> None:
    args = parse_args()

    # Bind CONFIG if present (keeps your logs intact in the wider project)
    if cfg_mod and hasattr(cfg_mod, "bind_modules"):
        try:
            cfg_mod.bind_modules(
                "main",
                "main_rsi",
                "optuna_imp",
                "optuna_imp_rsi",
                "backtest",
                "backtest_rsi",
            )
            print_kv(
                "CONFIG bound modules",
                "main, main_rsi, optuna_imp, optuna_imp_rsi, backtest, backtest_rsi",
            )
        except Exception:  # noqa: BLE001
            pass

    tkr = args.tkr.strip().upper()
    start_dt = parse_date(args.start)
    end_dt = parse_date(args.end)
    if end_dt < start_dt:
        raise SystemExit("end must be >= start")

    print_kv("Data download", f"{tkr} {start_dt.date()} → {end_dt.date()}")

    # Optimization (async; tolerates n_trials=0)
    print_kv("Starting optimization for MACD & RSI (async)…")
    macd_params, rsi_params = asyncio.run(
        run_optimizers_async(tkr, start_dt, end_dt, args.n_trials)
    )
    print_kv("MACD params", macd_params)
    print_kv("RSI params", rsi_params)

    # Last signal
    last_sig = compute_last_signal(tkr, start_dt, end_dt, macd_params, rsi_params)

    # Bullets (stub; extend/replace as needed)
    bullets = [
        BulletDTO(title="Window", body=f"{start_dt.date()} → {end_dt.date()}"),
        BulletDTO(title="Trials", body=str(args.n_trials)),
    ]

    # Chart (optional)
    chart_b64: Optional[str] = None
    chart_dto: Optional[ChartDTO] = None
    if args.include_chart:
        try:
            chart_b64 = render_chart_b64(tkr, start_dt, end_dt, args.layout)
            chart_dto = ChartDTO(format="png_base64", data=chart_b64)
        except Exception as e:  # noqa: BLE001
            log.exception("Chart rendering failed; continuing without chart: %s", e)

    # Build composite
    report = CompositeReportDTO(
        ticker=tkr,
        start_date=start_dt,
        end_date=end_dt,
        macd_params=macd_params,
        rsi_params=rsi_params,
        bullets=bullets,
        last_signal=last_sig,
        chart=chart_dto,
    )

    # JSON to STDOUT (keeps legacy workflows working)
    json_str = report.to_json(indent=2 if args.pretty else None)
    print(json_str)

    # Decide outputs
    out_json_path, out_html_path, out_png_path = _resolve_out_paths(args, chart_b64)

    # Write if any explicit output is requested
    if args.out or args.out_dir or args.out_json or args.out_html or args.out_png:
        out_json_path.write_text(json_str, encoding="utf-8")
        print_kv("Wrote composite report JSON", out_json_path)

        if out_png_path and chart_b64:
            _write_png_from_b64(out_png_path, chart_b64)
            print_kv("Wrote chart PNG", out_png_path)

        if out_html_path:
            html = _render_simple_html(report)
            out_html_path.write_text(html, encoding="utf-8")
            print_kv("Wrote HTML report", out_html_path)


if __name__ == "__main__":
    main()
