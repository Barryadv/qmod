from __future__ import annotations

# Standard library
import os

# --- Force headless Matplotlib before ANY matplotlib import (thread-safe) ---
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg", force=True)
matplotlib.rcParams["toolbar"] = "none"

# Standard library
import argparse
import asyncio
import base64
import io
import math, json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging

# Third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal

# Local config (new + legacy shim)
from config import AppConfig, init_CONFIG
import config as cfg_mod  # to access cfg_mod.CONFIG after init

# Legacy modules that read CONFIG (bind after init via helper)
import main as macd_main
import main_rsi as rsi_main
from paths import artifacts_dir
import optuna_imp as optuna_mod
import optuna_imp_rsi as optuna_rsi_mod
import backtest as backtest_mod
import backtest_rsi as backtest_rsi_mod

log = logging.getLogger(__name__)

# Project modules
from dto import MACDParamsDTO, RSIParamsDTO
from fetch import ohlcv



from dto import (
    ChartDTO,
    IndicatorDTO,
    SignalDTO,
    CompositeBulletDTO,
    CompositeReportDTO,
)

# ---- import the optimizers directly (cleaner than parsing subprocess stdout) ----
# main.py exposes: optimize_macd_from_config(...)
from main import optimize_macd_from_config  # type: ignore
# main_rsi.py exposes: optimize_rsi_from_config(...)
from main_rsi import optimize_rsi_from_config  # type: ignore

def _run_optimizers_sync(tkr: str, start_iso: str, end_iso: str, n_trials: int):
    """Run your async optimizer safely from sync code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        return loop.run_until_complete(run_optimizers_async(tkr, start_iso, end_iso, n_trials))
    return asyncio.run(run_optimizers_async(tkr, start_iso, end_iso, n_trials))

def _normalize_macd_params(p: Any) -> MACDParamsDTO:
    if isinstance(p, dict) and all(k in p for k in ("fast", "slow", "signal")):
        return MACDParamsDTO(fast=int(p["fast"]), slow=int(p["slow"]), signal=int(p["signal"]))
    raise ValueError(f"Invalid MACD params shape: {p!r}")

def _normalize_rsi_params(p: Any) -> RSIParamsDTO:
    if isinstance(p, dict) and all(k in p for k in ("length", "ob", "os")):
        return RSIParamsDTO(length=int(p["length"]), ob=int(p["ob"]), os=int(p["os"]))
    raise ValueError(f"Invalid RSI params shape: {p!r}")


def _extract_params_from_optimizer_result(opt: Any) -> tuple[MACDParamsDTO, RSIParamsDTO]:
    """
    Accepts several shapes:
      1) (macd_dict, rsi_dict, *rest)
      2) {"macd": {...}, "rsi": {...}} or {"macd_params": {...}, "rsi_params": {...}}
      3) path(s) to JSON file(s) containing the dicts
    Returns typed DTOs or raises ValueError with a detailed message.
    """
    log.debug("Optimizer raw result type=%s, value preview=%r", type(opt).__name__, opt if isinstance(opt, (dict, str)) else str(opt)[:200])

    # Case 1: tuple/list
    if isinstance(opt, (tuple, list)) and len(opt) >= 2:
        macd_raw, rsi_raw = opt[0], opt[1]
        return _normalize_macd_params(macd_raw), _normalize_rsi_params(rsi_raw)

    # Case 2: dict with various keys
    if isinstance(opt, dict):
        macd_raw = opt.get("macd") or opt.get("macd_params")
        rsi_raw  = opt.get("rsi")  or opt.get("rsi_params")
        if macd_raw is not None and rsi_raw is not None:
            return _normalize_macd_params(macd_raw), _normalize_rsi_params(rsi_raw)
        raise ValueError(f"Dict result missing macd/rsi keys: {list(opt.keys())}")

    # Case 3: file path to JSON with {"macd_params":{...}, "rsi_params":{...}}
    if isinstance(opt, str):
        p = Path(opt)
        if p.suffix.lower() == ".json" and p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                raise ValueError(f"Failed to parse optimizer JSON at {p}: {e}")
            macd_raw = data.get("macd") or data.get("macd_params")
            rsi_raw  = data.get("rsi")  or data.get("rsi_params")
            if macd_raw is not None and rsi_raw is not None:
                return _normalize_macd_params(macd_raw), _normalize_rsi_params(rsi_raw)
            raise ValueError(f"Optimizer JSON missing macd/rsi keys at {p}: {list(data.keys())}")

    # Unknown shape
    raise ValueError(f"Unsupported optimizer result type: {type(opt).__name__} -> {opt!r}")


# -----------------------
# Simple indicator helpers
# -----------------------
def bind_config_to_legacy_modules(cfg_dict) -> None:
    import importlib

    targets = {
        "main":            "MACD optimizer",
        "main_rsi":        "RSI optimizer",
        "optuna_imp":      "MACD optuna",
        "optuna_imp_rsi":  "RSI optuna",
        "backtest":        "MACD backtest",
        "backtest_rsi":    "RSI backtest",
    }

    bound = []
    skipped = []
    for modname, label in targets.items():
        try:
            mod = importlib.import_module(modname)  # import or fetch if already loaded
            setattr(mod, "CONFIG", cfg_dict)
            bound.append(modname)
        except Exception as e:
            skipped.append(f"{modname} ({label}) -> {type(e).__name__}: {e}")

    if bound:
        print_kv(f"CONFIG bound modules: {', '.join(bound)}")
    if skipped:
        print_kv("WARN: some modules not bound:\n  " + "\n  ".join(skipped))

def calc_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=close.index).ewm(alpha=1/length, adjust=False).mean()
    roll_down = pd.Series(loss, index=close.index).ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.bfill().fillna(50.0)

def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - macd_signal
    return macd, macd_signal, hist


# -----------------------
# Chart helpers
# -----------------------
def detect_macd_crosses(macd: pd.Series, signal: pd.Series) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Return (cross_up_idx, cross_down_idx) where MACD crosses its signal."""
    prev_above = macd.shift(1) > signal.shift(1)
    now_above = macd > signal
    cross_up = (~prev_above) & now_above
    cross_down = (prev_above) & (~now_above)
    return macd.index[cross_up.fillna(False)], macd.index[cross_down.fillna(False)]

def render_chart_b64(df: pd.DataFrame, ticker: str, macd_tuple, rsi: pd.Series) -> str:
    macd, macd_signal, hist = macd_tuple
    fig = plt.figure(figsize=(10, 7), dpi=120)

    # === Price ===
    ax_price = fig.add_subplot(3, 1, 1)
    ax_price.plot(df.index, df["Close"], linewidth=1.5, label="Close")

    # MACD cross markers on price
    up_idx, down_idx = detect_macd_crosses(macd, macd_signal)
    if len(up_idx):
        ax_price.scatter(up_idx, df.loc[up_idx, "Close"], marker="^", s=50, color="green", label="MACD Bullish ↑", zorder=3)
    if len(down_idx):
        ax_price.scatter(down_idx, df.loc[down_idx, "Close"], marker="v", s=50, color="red", label="MACD Bearish ↓", zorder=3)

    ax_price.set_title(f"{ticker} – Price")
    ax_price.grid(True, linestyle="--", alpha=0.3)
    ax_price.legend(loc="upper left", ncols=2, fontsize=8)

    # === MACD ===
    ax_macd = fig.add_subplot(3, 1, 2)
    ax_macd.plot(df.index, macd, linewidth=1.0, color="green", label="MACD")
    ax_macd.plot(df.index, macd_signal, linewidth=1.0, color="red", label="Signal")

    pos = hist.clip(lower=0)
    neg = hist.clip(upper=0)
    ax_macd.bar(df.index, pos, width=1.0, alpha=0.5, color="green", label="Hist +")
    ax_macd.bar(df.index, neg, width=1.0, alpha=0.5, color="red", label="Hist -")

    ax_macd.set_title("MACD")
    ax_macd.grid(True, linestyle="--", alpha=0.3)
    ax_macd.legend(loc="upper left", ncols=3, fontsize=8)

    # === RSI ===
    ax_rsi = fig.add_subplot(3, 1, 3)
    ax_rsi.plot(df.index, rsi, linewidth=1.0, label="RSI")
    ax_rsi.axhline(70, linestyle="--", linewidth=1.0, color="red")
    ax_rsi.axhline(30, linestyle="--", linewidth=1.0, color="green")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_title("RSI")
    ax_rsi.grid(True, linestyle="--", alpha=0.3)
    ax_rsi.legend(loc="upper left", fontsize=8)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# -----------------------
# Bullet generation
# -----------------------
def make_bullets(
    ticker: str,
    start: datetime,
    end: datetime,
    opt_start: datetime | None,
    opt_end: datetime | None,
    rsi_params: dict,
    macd_params: dict,
    last_rsi: float,
    last_macd: float,
    last_signal_line: float,
    signal_text: str,
) -> list[CompositeBulletDTO]:
    if opt_start and opt_end:
        p1 = (
            f"Technical picture for {ticker} from {start.date()} to {end.date()} "
            f"based on optimized MACD & RSI parameters from {opt_start.date()} to {opt_end.date()}."
        )
    else:
        p1 = (
            f"Technical picture for {ticker} from {start.date()} to {end.date()} "
            f"using current MACD & RSI parameters."
        )

    b1 = CompositeBulletDTO(title="Overview", body=p1)

    rsi_txt = (
        "RSI measures momentum on a 0–100 scale. "
        "Commonly: oversold < 30 (buy zone), overbought > 70 (sell zone). "
        f"Current parameters: length={rsi_params.get('length', 14)}, "
        f"overbought={rsi_params.get('ob', 70)}, oversold={rsi_params.get('os', 30)}."
    )
    b2 = CompositeBulletDTO(title="Reading RSI", body=rsi_txt)

    macd_txt = (
        "MACD tracks trend momentum via EMA differences. "
        "When MACD is above its signal line, momentum is strengthening; below it, weakening. "
        f"Current parameters: fast={macd_params.get('fast', 12)}, "
        f"slow={macd_params.get('slow', 26)}, signal={macd_params.get('signal', 9)}."
    )
    b3 = CompositeBulletDTO(title="Reading MACD", body=macd_txt)

    latest_txt = (
        f"As of {end.date()}: RSI={last_rsi:.1f}; MACD={last_macd:.4f} vs Signal={last_signal_line:.4f}. "
        f"Composite interpretation: {signal_text}."
    )
    b4 = CompositeBulletDTO(title="Latest Read", body=latest_txt)

    return [b1, b2, b3, b4]


# -----------------------
# Basic composite signal
# -----------------------
def infer_signal(last_rsi: float, macd: float, signal_line: float, ob: int, os: int) -> tuple[str, float, list[str]]:
    reasons = []
    score = 0.0

    if last_rsi <= os:
        score += 0.6
        reasons.append(f"RSI ≤ oversold ({last_rsi:.1f} ≤ {os})")
    elif last_rsi >= ob:
        score -= 0.6
        reasons.append(f"RSI ≥ overbought ({last_rsi:.1f} ≥ {ob})")
    else:
        reasons.append(f"RSI neutral ({last_rsi:.1f})")

    if macd > signal_line:
        score += 0.5
        reasons.append("MACD above signal (bullish momentum)")
    elif macd < signal_line:
        score -= 0.5
        reasons.append("MACD below signal (bearish momentum)")
    else:
        reasons.append("MACD roughly equal to signal")

    label = "buy" if score >= 0.4 else "sell" if score <= -0.4 else "hold"
    return label, score, reasons


# -----------------------
# ASYNC optimization orchestrator
# -----------------------
async def run_optimizers_async(
    tkr: str,
    start: str,
    end: str,
    n_trials: Optional[int]
) -> tuple[Dict[str, Any], Dict[str, Any], Path, Path]:
    """
    Run MACD & RSI optimizations concurrently in threads and return:
      (macd_params_norm, rsi_params_norm, macd_json_path, rsi_json_path)

    - Initializes config if needed (safe under FastAPI)
    - Normalizes MACD params -> {fast, slow, signal} (ints)
    - Normalizes RSI params  -> {length, os, ob} (ints)
    - Uses absolute artifact paths via artifacts_dir(...)
    """

    # Ensure CONFIG exists (Uvicorn doesn't run __main__)
    if getattr(cfg_mod, "CONFIG", None) is None:
        # try to initialize with provided range
        try:
            cfg_mod.init_CONFIG(ticker=tkr, start=start, end=end)
        except TypeError:
            # fallback if init signature differs
            cfg_mod.init_CONFIG()

    # Build a safe subdir name for fallbacks
    safe_tkr = tkr.replace(":", "_").replace("-", "_").replace("/", "_")

    def run_macd() -> tuple[Dict[str, Any], Path]:
        # Be generous with config keys to support legacy shapes
        res = macd_main.optimize_macd_from_config(
            cfg={"ticker": tkr, "start": start, "end": end, "start_date": start, "end_date": end},
            n_trials=n_trials,
            plot=False,
            vis_dir=None,
        )

        params = {
            "fast": int(res["params"]["fast"]),
            "slow": int(res["params"]["slow"]),
            "signal": int(res["params"]["signal"]),
        }

        # Prefer optimizer's vis_dir; otherwise place under artifacts/vis_macd/composite/<ticker>/
        vis_dir = Path(res.get("vis_dir") or artifacts_dir("vis_macd", "composite", safe_tkr))
        vis_dir.mkdir(parents=True, exist_ok=True)

        macd_json = vis_dir / "best_params.json"
        macd_json.write_text(json.dumps(params, indent=2), encoding="utf-8")
        return params, macd_json

    def run_rsi() -> tuple[Dict[str, Any], Path]:
        res = rsi_main.optimize_rsi_from_config(
            cfg={"ticker": tkr, "start": start, "end": end, "start_date": start, "end_date": end},
            n_trials=n_trials,
            plot=False,
            vis_dir=None,
        )

        rp = dict(res["params"])
        # Normalize to {length, os, ob}; accept legacy {period, lower, upper}
        if "period" in rp:
            params_norm = {
                "length": int(rp["period"]),
                "os": int(round(float(rp["lower"]))),
                "ob": int(round(float(rp["upper"]))),
            }
            rsi_save = {
                "period": params_norm["length"],
                "oversold": params_norm["os"],
                "overbought": params_norm["ob"],
            }
        else:
            params_norm = {
                "length": int(rp["length"]),
                "os": int(rp["os"]),
                "ob": int(rp["ob"]),
            }
            rsi_save = {
                "period": params_norm["length"],
                "oversold": params_norm["os"],
                "overbought": params_norm["ob"],
            }

        vis_dir = Path(res.get("vis_dir") or artifacts_dir("vis_rsi", "composite", safe_tkr))
        vis_dir.mkdir(parents=True, exist_ok=True)

        rsi_json = vis_dir / "best_params.json"
        rsi_json.write_text(json.dumps(rsi_save, indent=2), encoding="utf-8")
        return params_norm, rsi_json

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=2) as pool:
        macd_future = loop.run_in_executor(pool, run_macd)
        rsi_future  = loop.run_in_executor(pool, run_rsi)
        (macd_params, macd_json), (rsi_params, rsi_json) = await asyncio.gather(macd_future, rsi_future)

    return macd_params, rsi_params, macd_json, rsi_json

# -----------------------
# CLI / main
# -----------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build composite report DTO (bullets + chart).")
    p.add_argument("--tkr", "--ticker", dest="tkr", required=True, type=str)
    p.add_argument("--start", required=True, type=str, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, type=str, help="YYYY-MM-DD")

    # Optional: allow user to skip optimization and pass params manually
    p.add_argument("--macd-fast", type=int, default=None)
    p.add_argument("--macd-slow", type=int, default=None)
    p.add_argument("--macd-signal", type=int, default=None)
    p.add_argument("--rsi-length", type=int, default=None)
    p.add_argument("--rsi-ob", type=int, default=None)
    p.add_argument("--rsi-os", type=int, default=None)

    p.add_argument("--layout", type=str, default="two_column", choices=["two_column", "stacked"])
    p.add_argument("--opt-start", type=str, default=None)  # provenance-only
    p.add_argument("--opt-end", type=str, default=None)    # provenance-only
    p.add_argument("--n-trials", type=int, default=None, help="Optuna trials override for both optimizers")

    p.add_argument("--out", type=str, default=None, help="Write JSON to file")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    return p.parse_args()


def print_kv(msg: str) -> None:
    print(f"[INFO] {msg}")


def main() -> None:
    args = parse_args()
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    opt_start = datetime.fromisoformat(args.opt_start) if args.opt_start else None
    opt_end = datetime.fromisoformat(args.opt_end) if args.opt_end else None

    # ---------- Data download checkpoint ----------
    cfg = AppConfig(
        tkr=args.tkr,
        start=start.date().isoformat(),
        end=end.date().isoformat(),
    )

    # Populate legacy global CONFIG so main.py/main_rsi.py optimizers can read it
    init_CONFIG(argv=[
        "--tkr", args.tkr,
        "--start", args.start,
        "--end", args.end,
        *(["--n-trials", str(args.n_trials)] if args.n_trials else []),
    ])

    # Bind to every legacy module that references CONFIG
    bind_config_to_legacy_modules(cfg_mod.CONFIG)

    print_kv(f"Data download: {args.tkr} {args.start} → {args.end}")
    df, _benchmark = ohlcv(cfg, warmup_days=180)
    if df.empty:
        raise SystemExit("No data in the requested range.")
    print_kv(f"Data download complete. Rows={len(df)}")

    # Clip to requested display window (still use CLI args for explicit bounds)
    df = df.loc[(df.index >= pd.to_datetime(args.start)) & (df.index <= pd.to_datetime(args.end))].copy()

    # ---------- Optimization (async) OR manual params ----------
    macd_params: Dict[str, Any]
    rsi_params: Dict[str, Any]
    macd_json_path: Optional[Path] = None
    rsi_json_path: Optional[Path] = None

    manual_macd = (args.macd_fast is not None and args.macd_slow is not None and args.macd_signal is not None)
    manual_rsi = (args.rsi_length is not None and args.rsi_ob is not None and args.rsi_os is not None)

    if manual_macd and manual_rsi:
        macd_params = {"fast": args.macd_fast, "slow": args.macd_slow, "signal": args.macd_signal}
        rsi_params = {"length": args.rsi_length, "ob": args.rsi_ob, "os": args.rsi_os}
        print_kv(f"Optimization skipped (manual params provided).")
    else:
        print_kv("Starting optimization for MACD & RSI (async)…")
        macd_params, rsi_params, macd_json_path, rsi_json_path = asyncio.run(
            run_optimizers_async(args.tkr, args.start, args.end, args.n_trials)
        )
        print_kv(f"MACD optimized: fast={macd_params['fast']}, slow={macd_params['slow']}, signal={macd_params['signal']}")
        print_kv(f"RSI optimized: length={rsi_params['length']}, ob={rsi_params['ob']}, os={rsi_params['os']}")
        if macd_json_path:
            print_kv(f"MACD params saved: {macd_json_path}")
        if rsi_json_path:
            print_kv(f"RSI params saved:  {rsi_json_path}")

    # ---------- Build indicators/last signal ----------
    rsi_series = calc_rsi(df["Close"], length=int(rsi_params["length"]))
    macd_tuple = calc_macd(df["Close"], int(macd_params["fast"]), int(macd_params["slow"]), int(macd_params["signal"]))

    last_date = df.index[-1]
    last_rsi = float(rsi_series.iloc[-1])
    last_macd = float(macd_tuple[0].iloc[-1])
    last_signal_line = float(macd_tuple[1].iloc[-1])

    signal_label, score, reasons = infer_signal(last_rsi, last_macd, last_signal_line, int(rsi_params["ob"]), int(rsi_params["os"]))

    ind_list = [
        IndicatorDTO(name="RSI", params=rsi_params, values={"rsi": last_rsi}),
        IndicatorDTO(name="MACD", params=macd_params, values={"macd": last_macd, "signal": last_signal_line}),
    ]

    last_signal = SignalDTO(
        date=last_date.to_pydatetime(),
        ticker=args.tkr,
        signal=signal_label,
        score=round(score, 3),
        reasons=reasons,
    )

    chart_b64 = render_chart_b64(df, args.tkr, macd_tuple, rsi_series)
    chart_dto = ChartDTO(format="png_base64", data=chart_b64)

    bullets = make_bullets(
        ticker=args.tkr,
        start=start,
        end=end,
        opt_start=opt_start,
        opt_end=opt_end,
        rsi_params=rsi_params,
        macd_params=macd_params,
        last_rsi=last_rsi,
        last_macd=last_macd,
        last_signal_line=last_signal_line,
        signal_text=signal_label.upper(),
    )

    report = CompositeReportDTO(
        ticker=args.tkr,
        start_date=start,
        end_date=end,
        opt_start_date=opt_start,
        opt_end_date=opt_end,
        macd_params=macd_params,
        rsi_params=rsi_params,
        indicators=ind_list,
        layout=args.layout,
        bullets=bullets,
        chart=chart_dto,
        last_signal=last_signal,
    )

    json_str = report.to_json(indent=2 if args.pretty else None)
    print(json_str)
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json_str, encoding="utf-8")
        print_kv(f"Wrote composite report JSON: {outp}")

def build_composite_report(
    tkr: str,
    start: datetime,
    end: datetime,
    n_trials: int | None = None,
    layout: Literal["two_column", "stacked"] = "two_column",
) -> CompositeReportDTO:
    """
    Pure function: builds and returns a CompositeReportDTO (no prints/files).
    Robust to optimizer returning None/bad shapes. Uses typed param DTOs.
    """
    # Initialize legacy CONFIG so existing modules (optimizers/fetch) read it
    cfg = AppConfig(tkr=tkr, start=start.date().isoformat(), end=end.date().isoformat())
    init_CONFIG(argv=[
        "--tkr", tkr,
        "--start", start.date().isoformat(),
        "--end", end.date().isoformat(),
        *(["--n-trials", str(n_trials)] if n_trials else []),
    ])

    # Fetch OHLCV (with warmup)
    df, _benchmark = ohlcv(cfg, warmup_days=180)
    df = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))].copy()
    if df.empty:
        raise ValueError("No data in the requested range.")

    # --- Params: either optimize or use defaults (typed DTOs) ---
    if n_trials and n_trials > 0:
        opt = _run_optimizers_sync(tkr, start.date().isoformat(), end.date().isoformat(), n_trials)
        if opt is None:
            log.error("Optimizer returned None")
            # Strict path (400): raise ValueError("Optimizer returned no parameters")
            # Fallback path (200): use defaults
            macd_params = MACDParamsDTO(fast=12, slow=26, signal=9)
            rsi_params  = RSIParamsDTO(length=14, ob=70, os=30)
        else:
            try:
                macd_params, rsi_params = _extract_params_from_optimizer_result(opt)
            except Exception as e:
                log.warning("Optimization failed: %s. Falling back to defaults.", e)
                macd_params = MACDParamsDTO(fast=12, slow=26, signal=9)
                rsi_params  = RSIParamsDTO(length=14, ob=70, os=30)
    else:
        macd_params = MACDParamsDTO(fast=12, slow=26, signal=9)
        rsi_params  = RSIParamsDTO(length=14, ob=70, os=30)

    # --- Indicators using typed params ---
    rsi_series = calc_rsi(df["Close"], length=int(rsi_params.length))
    macd_tuple = calc_macd(
        df["Close"],
        int(macd_params.fast), int(macd_params.slow), int(macd_params.signal)
    )

    last_date = df.index[-1].to_pydatetime()
    last_rsi = float(rsi_series.iloc[-1])
    last_macd = float(macd_tuple[0].iloc[-1])
    last_signal_line = float(macd_tuple[1].iloc[-1])

    signal_label, score, reasons = infer_signal(
        last_rsi, last_macd, last_signal_line,
        int(rsi_params.ob), int(rsi_params.os)
    )

    indicators = [
        IndicatorDTO(name="RSI",  params=rsi_params.model_dump(),  values={"rsi": last_rsi}),
        IndicatorDTO(name="MACD", params=macd_params.model_dump(), values={"macd": last_macd, "signal": last_signal_line}),
    ]

    # Chart (base64)
    chart_b64 = render_chart_b64(df, tkr, macd_tuple, rsi_series)
    chart = ChartDTO(format="png_base64", data=chart_b64)

    bullets = [
        CompositeBulletDTO(title="Composite interpretation", body=signal_label.upper()),
        CompositeBulletDTO(title="RSI (last)", body=f"{last_rsi:.1f}"),
        CompositeBulletDTO(title="MACD vs Signal (last)", body=f"{last_macd:.4f} vs {last_signal_line:.4f}"),
    ]

    return CompositeReportDTO(
        ticker=tkr,
        start_date=start,
        end_date=end,
        macd_params=macd_params,   # typed DTO
        rsi_params=rsi_params,     # typed DTO
        indicators=indicators,
        layout=layout,
        bullets=bullets,
        chart=chart,
        last_signal=SignalDTO(
            date=last_date,
            ticker=tkr,
            signal=signal_label,
            score=round(score, 3),
            reasons=reasons,
        ),
    )

if __name__ == "__main__":
    main()
