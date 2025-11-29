# qmod/api.py — FastAPI server for qmod pipeline
"""
FastAPI application exposing the qmod analysis pipeline via HTTP.

Start with:
    python -m qmod --serve
    python -m qmod --serve --port 8000 --host 0.0.0.0

Or directly:
    uvicorn qmod.api:app --reload
"""
from __future__ import annotations

import traceback
from datetime import datetime
from logging.config import dictConfig
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, HTMLResponse

from qmod.dto import RunOutputDTO, ErrorDTO
from qmod.pipeline import run_once, run_with_report, run_optimization, DEFAULT_MACD, DEFAULT_RSI

# ---- Logging Configuration ----
dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(asctime)s %(levelname)s [%(name)s] %(message)s"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "default"},
    },
    "root": {"level": "INFO", "handlers": ["console"]},
    "loggers": {
        "qmod": {"level": "INFO", "handlers": ["console"], "propagate": False},
        "uvicorn": {"level": "INFO", "handlers": ["console"], "propagate": False},
    },
})

# ---- FastAPI App ----
app = FastAPI(
    title="qmod API",
    version="1.0.0",
    description="Quantitative analysis pipeline with MACD/RSI optimization, signals, and reporting.",
    openapi_tags=[
        {"name": "Analysis", "description": "Run analysis and optimization on tickers."},
        {"name": "Reports", "description": "Generate HTML reports."},
        {"name": "Health", "description": "Service status."},
        {"name": "Meta", "description": "Schema and tooling."},
    ],
)

# CORS middleware for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Helper Functions ----
def _parse_date(date_str: str) -> datetime:
    """Parse date string to datetime."""
    return datetime.strptime(date_str, "%Y-%m-%d")


def _build_config_override(
    tkr: str,
    start: Optional[datetime],
    end: Optional[datetime],
) -> dict:
    """Build config override dict from query params."""
    overrides = {"tkr": tkr.upper()}
    if start:
        overrides["start"] = start.strftime("%Y-%m-%d")
    if end:
        overrides["end"] = end.strftime("%Y-%m-%d")
    return overrides


# ---- Endpoints ----

@app.get(
    "/analyze",
    operation_id="analyzeStock",
    tags=["Analysis"],
    summary="Run analysis on a ticker",
    response_model=RunOutputDTO,
    responses={
        200: {"description": "Analysis completed successfully"},
        400: {"description": "Bad request", "model": ErrorDTO},
        500: {"description": "Unexpected error", "model": ErrorDTO},
    },
)
def analyze(
    tkr: str = Query(..., description="Ticker symbol (e.g., AAPL, GM, BTC-USD)"),
    start: Optional[datetime] = Query(None, description="Start date (YYYY-MM-DD). Defaults to 1 year ago."),
    end: Optional[datetime] = Query(None, description="End date (YYYY-MM-DD). Defaults to today."),
    n_trials: int = Query(0, ge=0, le=500, description="Optuna optimization trials. 0 = use defaults."),
    optimizer: Literal["macd", "rsi", "both"] = Query("macd", description="Which optimizer to run."),
) -> RunOutputDTO:
    """
    Run the qmod analysis pipeline on a ticker.
    
    Returns signal (buy/sell/hold), score, indicators, and optionally optimized parameters.
    """
    # Validate date range
    if start and end and end <= start:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDTO(
                code="INVALID_DATE_RANGE",
                message="end date must be after start date",
                details={"start": start.isoformat(), "end": end.isoformat()},
            ).model_dump(),
        )

    try:
        config_override = _build_config_override(tkr, start, end)
        result = run_once(
            config_override=config_override,
            n_trials=n_trials,
            optimizer=optimizer,
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDTO(
                code="DOMAIN_ERROR",
                message=str(e),
                details={},
            ).model_dump(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDTO(
                code="UNEXPECTED_ERROR",
                message=str(e),
                details={"trace": traceback.format_exc()},
            ).model_dump(),
        )


@app.get(
    "/report",
    operation_id="generateReport",
    tags=["Reports"],
    summary="Generate HTML report for a ticker",
    response_class=HTMLResponse,
    responses={
        200: {"description": "HTML report generated", "content": {"text/html": {}}},
        400: {"description": "Bad request", "model": ErrorDTO},
        500: {"description": "Unexpected error", "model": ErrorDTO},
    },
)
def generate_report(
    tkr: str = Query(..., description="Ticker symbol (e.g., AAPL, GM, BTC-USD)"),
    start: Optional[datetime] = Query(None, description="Start date (YYYY-MM-DD). Defaults to 1 year ago."),
    end: Optional[datetime] = Query(None, description="End date (YYYY-MM-DD). Defaults to today."),
    n_trials: int = Query(0, ge=0, le=500, description="Optuna optimization trials. 0 = use defaults."),
    optimizer: Literal["macd", "rsi", "both"] = Query("macd", description="Which optimizer to run."),
) -> HTMLResponse:
    """
    Run analysis and generate an HTML composite report.
    
    Returns the HTML content directly (viewable in browser).
    """
    # Validate date range
    if start and end and end <= start:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDTO(
                code="INVALID_DATE_RANGE",
                message="end date must be after start date",
                details={"start": start.isoformat(), "end": end.isoformat()},
            ).model_dump(),
        )

    try:
        config_override = _build_config_override(tkr, start, end)
        _result, report_path = run_with_report(
            config_override=config_override,
            n_trials=n_trials,
            optimizer=optimizer,
        )
        
        # Read and return the HTML content
        html_content = Path(report_path).read_text(encoding="utf-8")
        return HTMLResponse(content=html_content)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDTO(
                code="DOMAIN_ERROR",
                message=str(e),
                details={},
            ).model_dump(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDTO(
                code="UNEXPECTED_ERROR",
                message=str(e),
                details={"trace": traceback.format_exc()},
            ).model_dump(),
        )


@app.get(
    "/report/json",
    operation_id="generateReportJson",
    tags=["Reports"],
    summary="Generate report and return JSON with file path",
    response_model=dict,
    responses={
        200: {"description": "Report generated, path returned"},
        400: {"description": "Bad request", "model": ErrorDTO},
        500: {"description": "Unexpected error", "model": ErrorDTO},
    },
)
def generate_report_json(
    tkr: str = Query(..., description="Ticker symbol (e.g., AAPL, GM, BTC-USD)"),
    start: Optional[datetime] = Query(None, description="Start date (YYYY-MM-DD). Defaults to 1 year ago."),
    end: Optional[datetime] = Query(None, description="End date (YYYY-MM-DD). Defaults to today."),
    n_trials: int = Query(0, ge=0, le=500, description="Optuna optimization trials. 0 = use defaults."),
    optimizer: Literal["macd", "rsi", "both"] = Query("macd", description="Which optimizer to run."),
) -> dict:
    """
    Run analysis, generate HTML report, and return the analysis result with file path.
    """
    # Validate date range
    if start and end and end <= start:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDTO(
                code="INVALID_DATE_RANGE",
                message="end date must be after start date",
                details={"start": start.isoformat(), "end": end.isoformat()},
            ).model_dump(),
        )

    try:
        config_override = _build_config_override(tkr, start, end)
        result, report_path = run_with_report(
            config_override=config_override,
            n_trials=n_trials,
            optimizer=optimizer,
        )
        
        return {
            "analysis": result.model_dump(),
            "report_path": str(report_path),
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorDTO(
                code="DOMAIN_ERROR",
                message=str(e),
                details={},
            ).model_dump(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorDTO(
                code="UNEXPECTED_ERROR",
                message=str(e),
                details={"trace": traceback.format_exc()},
            ).model_dump(),
        )


@app.get(
    "/defaults",
    operation_id="getDefaults",
    tags=["Meta"],
    summary="Get default MACD and RSI parameters",
)
def get_defaults() -> dict:
    """Return the default MACD and RSI parameters used when optimization is skipped."""
    return {
        "macd": DEFAULT_MACD,
        "rsi": DEFAULT_RSI,
    }


@app.get(
    "/healthz",
    operation_id="getHealth",
    tags=["Health"],
    summary="Health check / liveness probe",
)
def healthz() -> dict:
    """Simple health check for container orchestration."""
    return {"ok": True, "service": "qmod"}


@app.get(
    "/schema",
    operation_id="getOpenAPISchema",
    tags=["Meta"],
    summary="Get OpenAPI schema",
)
def schema() -> JSONResponse:
    """Expose the OpenAPI schema (useful for codegen and LLM tools)."""
    return JSONResponse(
        get_openapi(title=app.title, version=app.version, routes=app.routes)
    )


# ---- Server Runner (for direct execution) ----
def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the FastAPI server using uvicorn."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()

