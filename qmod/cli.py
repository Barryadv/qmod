# qmod/cli.py
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from qmod.pipeline import run_once, run_visual_workflow, run_with_report

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _parse_date(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    try:
        datetime.strptime(s, "%Y-%m-%d")
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid date '{s}'. Use YYYY-MM-DD.") from e
    return s


def _bool_env_style(v: Optional[str]) -> Optional[bool]:
    if v is None:
        return None
    s = v.strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value '{v}'.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qmod",
        description="QuantModule CLI — run the pipeline once, emit JSON, or start API server."
    )

    # Server mode
    p.add_argument(
        "--serve",
        action="store_true",
        help="Start FastAPI server instead of running analysis.",
    )
    p.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server (default: 127.0.0.1). Use 0.0.0.0 for external access.",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000).",
    )

    # Core run parameters
    p.add_argument("--tkr", "--ticker", dest="tkr", type=str, help="Ticker, e.g., AAPL")
    p.add_argument("--start", type=_parse_date, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", type=_parse_date, help="End date (YYYY-MM-DD)")

    # Optimization parameters
    p.add_argument(
        "--n-trials",
        dest="n_trials",
        type=int,
        default=0,
        help="Optuna optimization trials (0 = use defaults, >0 = run optimizer)",
    )
    p.add_argument(
        "--optimizer",
        dest="optimizer",
        type=str,
        default="macd",
        choices=["macd", "rsi", "both"],
        help="Which optimizer to run: macd (default), rsi, or both",
    )

    # Optional toggles
    p.add_argument("--store-charts-in-json", dest="store_charts_in_json",
                   type=_bool_env_style, metavar="BOOL",
                   help="Include base64 PNG chart in JSON (true/false).")
    p.add_argument("--disable-db", dest="disable_db",
                   type=_bool_env_style, metavar="BOOL",
                   help="Disable DB interactions (true/false).")
    p.add_argument("--disable-api", dest="disable_api",
                   type=_bool_env_style, metavar="BOOL",
                   help="Disable external API calls (true/false).")

    # Visual-only workflow
    p.add_argument("--visual-only", action="store_true",
                   help="Run Workflow 1 (ratio vs benchmark + 50D SMA) and print chart JSON only.")

    # HTML report generation
    p.add_argument(
        "--report",
        action="store_true",
        help="Generate an HTML composite report (saved to artifacts/reports/<ticker>/)",
    )
    p.add_argument(
        "--report-out",
        dest="report_out",
        type=Path,
        help="Custom path for the HTML report file.",
    )

    # Output controls
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    p.add_argument("--out", type=Path, help="Write JSON output to file instead of stdout.")

    # Logging
    p.add_argument("--log-level", default="INFO",
                   choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
                   help="Logging level (default: INFO).")

    return p


def _clean_overrides(ns: argparse.Namespace) -> Dict[str, Any]:
    raw: Dict[str, Any] = vars(ns).copy()
    # Remove CLI-only args that aren't config overrides
    cli_only = [
        "pretty", "out", "log_level", "visual_only", 
        "n_trials", "optimizer", "report", "report_out",
        "serve", "host", "port",
    ]
    for k in cli_only:
        raw.pop(k, None)
    return {k: v for k, v in raw.items() if v is not None}


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Server mode: start FastAPI
    if args.serve:
        try:
            import uvicorn
            from qmod.api import app
        except ImportError as e:
            logger.error("Failed to import server dependencies: %s", e)
            logger.error("Make sure fastapi and uvicorn are installed: pip install fastapi uvicorn")
            return
        
        print(f"Starting qmod API server at http://{args.host}:{args.port}")
        print(f"API docs available at http://{args.host}:{args.port}/docs")
        print("Press Ctrl+C to stop.")
        uvicorn.run(app, host=args.host, port=args.port)
        return

    overrides = _clean_overrides(args)
    logger.debug("Config overrides: %s", overrides)

    report_path = None
    
    if args.visual_only:
        chart_dto, _bench = run_visual_workflow(config_override=overrides)
        payload = chart_dto.model_dump_json(indent=2) if args.pretty else chart_dto.model_dump_json()
    elif args.report:
        # Run pipeline with HTML report generation
        result, report_path = run_with_report(
            config_override=overrides,
            n_trials=args.n_trials,
            optimizer=args.optimizer,
            report_out=args.report_out,
        )
        payload = result.model_dump_json(indent=2) if args.pretty else result.model_dump_json()
    else:
        # Pass optimization parameters to the pipeline
        result = run_once(
            config_override=overrides,
            n_trials=args.n_trials,
            optimizer=args.optimizer,
        )
        payload = result.model_dump_json(indent=2) if args.pretty else result.model_dump_json()

    if report_path:
        print(f"HTML Report: {report_path}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"Wrote: {args.out}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
