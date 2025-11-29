# qmod — Quantitative Analysis Pipeline

A Python-based quantitative analysis toolkit for stock market technical indicators with **Optuna hyperparameter optimization**, **FastAPI server**, and **HTML report generation**.

## Features

- 📊 **Technical Indicators**: MACD, RSI, Bollinger Bands, SMA/EMA
- 🔧 **Hyperparameter Optimization**: Optuna-powered optimization for MACD and RSI parameters
- 🌐 **FastAPI Server**: REST API for programmatic access
- 📄 **HTML Reports**: Two-column composite reports with interactive charts
- 💾 **Timestamped Artifacts**: All outputs are timestamped to preserve history

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Barryadv/qmod.git
cd qmod

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### CLI Usage

```bash
# Basic analysis (uses default parameters)
python -m qmod --tkr AAPL --start 2024-01-01 --end 2024-06-30

# With MACD optimization (10 trials)
python -m qmod --tkr GM --start 2024-01-01 --end 2024-06-30 --n-trials 10 --optimizer macd

# With RSI optimization
python -m qmod --tkr MSFT --n-trials 15 --optimizer rsi

# Optimize both MACD and RSI
python -m qmod --tkr AAPL --n-trials 10 --optimizer both

# Generate HTML report
python -m qmod --tkr GM --n-trials 10 --optimizer rsi --report

# Pretty-print JSON output
python -m qmod --tkr AAPL --pretty
```

### Start the API Server

```bash
# Start server on default port (8000)
python -m qmod --serve

# Custom host and port
python -m qmod --serve --host 0.0.0.0 --port 8080
```

Once running, access:
- **Interactive API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/healthz
- **Analyze Endpoint**: http://localhost:8000/analyze?tkr=AAPL&n_trials=5

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | GET | Run analysis on a ticker, return JSON |
| `/report` | GET | Generate and return HTML report |
| `/report/json` | GET | Generate report, return JSON with file path |
| `/defaults` | GET | Get default MACD/RSI parameters |
| `/healthz` | GET | Health check for container orchestration |
| `/docs` | GET | Interactive Swagger UI documentation |

### Example API Call

```bash
curl "http://localhost:8000/analyze?tkr=GM&start=2024-01-01&end=2024-06-30&n_trials=5&optimizer=rsi"
```

---

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--tkr` | Ticker symbol (e.g., AAPL, GM, BTC-USD) | Required |
| `--start` | Start date (YYYY-MM-DD) | 1 year ago |
| `--end` | End date (YYYY-MM-DD) | Today |
| `--n-trials` | Optuna optimization trials (0 = use defaults) | 0 |
| `--optimizer` | Which optimizer: `macd`, `rsi`, or `both` | macd |
| `--report` | Generate HTML composite report | False |
| `--serve` | Start FastAPI server | False |
| `--host` | Server bind address | 127.0.0.1 |
| `--port` | Server port | 8000 |
| `--pretty` | Pretty-print JSON output | False |

---

## Module Overview

| Module | Description |
|--------|-------------|
| `pipeline.py` | Core analysis pipeline orchestrating data fetch, optimization, and signal generation |
| `cli.py` | Command-line interface with argument parsing |
| `api.py` | FastAPI server with REST endpoints |
| `dto.py` | Pydantic data transfer objects for type-safe data handling |
| `optuna_imp.py` | Optuna-based MACD parameter optimization |
| `optuna_imp_rsi.py` | Optuna-based RSI parameter optimization |
| `report_generator.py` | HTML report generation with embedded charts |
| `indicators.py` | Technical indicator calculations (MACD, RSI, Bollinger, SMA/EMA) |
| `signals.py` | Trading signal generation based on indicator values |
| `patterns.py` | Pattern detection (moving average crossovers) |
| `data_importer.py` | Market data fetching via yfinance |
| `charting.py` | Chart generation with matplotlib |
| `config.py` | Application configuration via Pydantic Settings |
| `paths.py` | Artifact directory management |
| `db.py` | SQLAlchemy database integration |
| `backtest.py` | MACD backtesting module |
| `backtest_rsi.py` | RSI backtesting module |

---

## Output Structure

All artifacts are saved with timestamps to prevent overwrites:

```
artifacts/
├── optimized/
│   └── AAPL/
│       ├── AAPL_20241129_143052_optimized_params.json
│       ├── macd/
│       │   ├── 20241129_143052_best_params.json
│       │   └── 20241129_143052_study_summary.json
│       └── rsi/
│           └── ...
└── reports/
    └── AAPL/
        └── AAPL_20241129_143052_report.html
```

---

## Default Parameters

When optimization is skipped (`--n-trials 0`):

| Indicator | Parameter | Default |
|-----------|-----------|---------|
| MACD | Fast | 12 |
| MACD | Slow | 26 |
| MACD | Signal | 9 |
| RSI | Length | 14 |
| RSI | Oversold | 30 |
| RSI | Overbought | 70 |

---

## Requirements

- Python 3.10+
- pandas, numpy, matplotlib
- yfinance (market data)
- optuna (optimization)
- pydantic, pydantic-settings (configuration)
- fastapi, uvicorn (API server)
- SQLAlchemy (database)

See `requirements.txt` for full list.

---

## License

MIT License

---

## Author

Barry Ehrlich — [github.com/Barryadv](https://github.com/Barryadv)

