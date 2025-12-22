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

import glob
import os
import re
import traceback
from datetime import datetime
from logging.config import dictConfig
from pathlib import Path
from typing import Literal, Optional

import httpx
from fastapi import FastAPI, Query, HTTPException, status, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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

# ---- Static Files & Frontend ----
# Determine the path to the frontend directory
_FRONTEND_DIR = Path(__file__).parent / "frontend"

# Serve static files from frontend directory (if it exists)
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_frontend():
    """Serve the frontend index.html at root path."""
    index_path = _FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return HTMLResponse("<h1>qmod API</h1><p>Visit <a href='/docs'>/docs</a> for API documentation.</p>")


# ---- DivArist Section ----
# Path to the DivArist project (for local development)
_DIVARIST_DIR = Path(os.environ.get("DIVARIST_PATH", ""))
if not _DIVARIST_DIR.exists():
    # Fallback to sibling directory for local development
    _DIVARIST_DIR = Path(__file__).parent.parent.parent / "DivArist"

# File mapping for DivArist downloads
# Each entry: (env_var_name, local_subdir, local_pattern, media_type)
_DIVARIST_FILES = {
    "ew_portfolio_report": ("DIVARIST_EW_PORTFOLIO_REPORT", "", "EW_Portfolio_Report_*.html", "text/html"),
    "dashboard_train": ("DIVARIST_DASHBOARD_TRAIN", "Dashboards", "dashboard_train_*.html", "text/html"),
    "dashboard_test": ("DIVARIST_DASHBOARD_TEST", "Dashboards", "dashboard_test_*.html", "text/html"),
    "backtest_chart": ("DIVARIST_BACKTEST_CHART", "ScenarioGrid", "backtest_chart_*.png", "image/png"),
    "backtest_heatmap": ("DIVARIST_BACKTEST_HEATMAP", "ScenarioGrid", "backtest_heatmap_*.png", "image/png"),
    "simulation_train": ("DIVARIST_SIMULATION_TRAIN", "SimulationTest", "simulation_train_*.png", "image/png"),
    "simulation_test": ("DIVARIST_SIMULATION_TEST", "SimulationTest", "simulation_test_*.png", "image/png"),
    "readme": ("DIVARIST_README_URL", "", "README.md", "text/markdown"),
}

# README can also be a Dropbox URL
_DIVARIST_README_URL = os.environ.get("DIVARIST_README_URL", "")


def _get_dropbox_url(file_id: str) -> Optional[str]:
    """Get Dropbox URL for a file from environment variable."""
    if file_id not in _DIVARIST_FILES:
        return None
    env_var = _DIVARIST_FILES[file_id][0]
    url = os.environ.get(env_var, "")
    if url:
        # Convert Dropbox share URL to direct download URL
        # ?dl=0 -> ?dl=1 for direct download
        if "dropbox.com" in url and "?dl=0" in url:
            url = url.replace("?dl=0", "?dl=1")
        elif "dropbox.com" in url and "?dl=" not in url:
            url = url + ("&dl=1" if "?" in url else "?dl=1")
    return url if url else None


def _find_latest_file(subdir: str, pattern: str) -> Optional[Path]:
    """Find the latest file matching a pattern in a DivArist subdirectory."""
    if subdir:
        search_path = _DIVARIST_DIR / subdir / pattern
    else:
        # Root level file
        search_path = _DIVARIST_DIR / pattern
    
    files = glob.glob(str(search_path))
    if not files:
        # Try non-timestamped version
        root_pattern = pattern.replace("_*", "").replace("*", "")
        if subdir:
            root_file = _DIVARIST_DIR / subdir / root_pattern
        else:
            root_file = _DIVARIST_DIR / root_pattern
        if root_file.exists():
            return root_file
        # Also try just the base name
        if subdir:
            base_file = _DIVARIST_DIR / subdir / (root_pattern + ".html" if "html" in pattern else root_pattern + ".png")
        else:
            base_file = _DIVARIST_DIR / (root_pattern + ".html" if "html" in pattern else root_pattern + ".png")
        if base_file.exists():
            return base_file
        return None
    # Return the most recently modified file
    return Path(max(files, key=os.path.getmtime))


def _markdown_to_html(md_text: str) -> str:
    """Convert markdown to basic HTML (simple implementation)."""
    html = md_text
    
    # Escape HTML special chars first (except for our conversions)
    # Skip this to allow tables to work
    
    # Headers
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    
    # Bold and italic
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
    
    # Inline code
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
    
    # Code blocks
    html = re.sub(r'```(\w*)\n(.*?)```', r'<pre><code>\2</code></pre>', html, flags=re.DOTALL)
    
    # Blockquotes
    html = re.sub(r'^> (.+)$', r'<blockquote>\1</blockquote>', html, flags=re.MULTILINE)
    
    # Horizontal rules
    html = re.sub(r'^---+$', r'<hr>', html, flags=re.MULTILINE)
    
    # Tables (basic support)
    lines = html.split('\n')
    in_table = False
    table_lines = []
    result_lines = []
    
    for line in lines:
        if '|' in line and line.strip().startswith('|'):
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(line)
        else:
            if in_table:
                # Process accumulated table
                result_lines.append(_convert_table(table_lines))
                in_table = False
                table_lines = []
            result_lines.append(line)
    
    if in_table:
        result_lines.append(_convert_table(table_lines))
    
    html = '\n'.join(result_lines)
    
    # Lists
    html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'(<li>.*</li>\n?)+', r'<ul>\g<0></ul>', html)
    
    # Paragraphs (lines that aren't already tagged)
    lines = html.split('\n')
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('<') and not stripped.startswith('|'):
            result.append(f'<p>{stripped}</p>')
        else:
            result.append(line)
    
    return '\n'.join(result)


def _convert_table(lines: list) -> str:
    """Convert markdown table lines to HTML table."""
    if len(lines) < 2:
        return '\n'.join(lines)
    
    html = ['<table>']
    for i, line in enumerate(lines):
        if '---' in line:
            continue  # Skip separator row
        cells = [c.strip() for c in line.split('|')[1:-1]]
        tag = 'th' if i == 0 else 'td'
        row = '<tr>' + ''.join(f'<{tag}>{c}</{tag}>' for c in cells) + '</tr>'
        html.append(row)
    html.append('</table>')
    return '\n'.join(html)


class ChatRequest(BaseModel):
    question: str


@app.get("/divarist", include_in_schema=False)
def serve_divarist():
    """Serve the DivArist landing page."""
    divarist_path = _FRONTEND_DIR / "divarist.html"
    if divarist_path.exists():
        return FileResponse(str(divarist_path), media_type="text/html")
    return HTMLResponse("<h1>DivArist</h1><p>Page not found.</p>")


@app.get("/api/divarist/readme", tags=["DivArist"])
async def get_divarist_readme():
    """Get the DivArist README as HTML."""
    md_content = None
    
    # Try Dropbox URL first
    if _DIVARIST_README_URL:
        try:
            url = _DIVARIST_README_URL
            if "dropbox.com" in url and "?dl=0" in url:
                url = url.replace("?dl=0", "?dl=1")
            elif "dropbox.com" in url and "?dl=" not in url:
                url = url + ("&dl=1" if "?" in url else "?dl=1")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, follow_redirects=True)
                if response.status_code == 200:
                    md_content = response.text
        except Exception:
            pass  # Fall through to local file
    
    # Fallback to local file
    if not md_content:
        readme_path = _DIVARIST_DIR / "README.md"
        if readme_path.exists():
            md_content = readme_path.read_text(encoding="utf-8")
    
    if not md_content:
        raise HTTPException(status_code=404, detail="README not found")
    
    html_content = _markdown_to_html(md_content)
    return {"html": html_content, "markdown": md_content}


@app.get("/api/divarist/file/{file_id}", tags=["DivArist"])
async def get_divarist_file(file_id: str):
    """Download a DivArist output file. Supports Dropbox URLs or local files."""
    if file_id not in _DIVARIST_FILES:
        raise HTTPException(status_code=404, detail=f"Unknown file: {file_id}")
    
    env_var, subdir, pattern, media_type = _DIVARIST_FILES[file_id]
    
    # Check for Dropbox URL first
    dropbox_url = _get_dropbox_url(file_id)
    if dropbox_url:
        # Redirect to Dropbox for direct download
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=dropbox_url, status_code=302)
    
    # Fallback to local file
    if subdir:
        file_path = _find_latest_file(subdir, pattern)
    else:
        # Root level file - handle both exact matches and glob patterns
        if "*" in pattern:
            # Use glob pattern matching for timestamped files
            file_path = _find_latest_file("", pattern)
        else:
            # Exact file match (e.g., README.md)
            file_path = _DIVARIST_DIR / pattern
            if not file_path.exists():
                file_path = None
    
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
    
    return FileResponse(
        str(file_path),
        media_type=media_type,
        filename=file_path.name
    )


@app.post("/api/divarist/chat", tags=["DivArist"])
async def divarist_chat(request: ChatRequest):
    """Chat about the DivArist strategy using LLM (Anthropic or OpenAI)."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Load README for context - try Dropbox first, then local
    readme_content = None
    
    if _DIVARIST_README_URL:
        try:
            url = _DIVARIST_README_URL
            if "dropbox.com" in url and "?dl=0" in url:
                url = url.replace("?dl=0", "?dl=1")
            elif "dropbox.com" in url and "?dl=" not in url:
                url = url + ("&dl=1" if "?" in url else "?dl=1")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, follow_redirects=True)
                if response.status_code == 200:
                    readme_content = response.text
        except Exception:
            pass
    
    if not readme_content:
        readme_path = _DIVARIST_DIR / "README.md"
        if readme_path.exists():
            readme_content = readme_path.read_text(encoding="utf-8")
        else:
            readme_content = "Documentation not available."
    
    # Build system prompt
    system_prompt = f"""You are an expert assistant for the DivArist project - an Emerging Market Dividend Aristocrats investment strategy.

You have access to the full project documentation below. Answer questions accurately based on this documentation.

FORMATTING RULES (STRICT):
- Keep responses under 500 characters
- Use 3-5 bullet points maximum
- Each bullet should be 1-2 sentences
- No headers, no markdown formatting except bullets
- Start directly with bullets, no intro text
- If the answer isn't in the documentation, say so briefly

=== PROJECT DOCUMENTATION ===
{readme_content[:15000]}
=== END DOCUMENTATION ==="""

    # Try Anthropic first
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    if anthropic_key:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": anthropic_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 400,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": question}],
                    },
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["content"][0]["text"]
                    return {"answer": answer, "provider": "anthropic"}
        except Exception as e:
            # Fall through to OpenAI
            pass
    
    # Fallback to OpenAI
    if openai_key:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openai_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "gpt-4o",
                        "max_tokens": 400,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question},
                        ],
                    },
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["choices"][0]["message"]["content"]
                    return {"answer": answer, "provider": "openai"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")
    
    raise HTTPException(
        status_code=503,
        detail="No API keys configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable."
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
    optimizer: Literal["macd", "rsi", "both", "combined"] = Query("macd", description="Which optimizer to run."),
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
    optimizer: Literal["macd", "rsi", "both", "combined"] = Query("macd", description="Which optimizer to run."),
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
    optimizer: Literal["macd", "rsi", "both", "combined"] = Query("macd", description="Which optimizer to run."),
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

