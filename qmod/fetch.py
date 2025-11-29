# qmod/fetch.py
from typing import Tuple
import pandas as pd
from .config import AppConfig
from .pipeline import fetch_with_benchmark

def ohlcv(cfg: AppConfig, warmup_days: int = 60) -> Tuple[pd.DataFrame, str]:
    """Return primary OHLCV DataFrame and benchmark symbol."""
    return fetch_with_benchmark(cfg, warmup_days=warmup_days)
