from __future__ import annotations

import logging
from typing import Any, Dict, Union

import pandas as pd
import yfinance as yf
from datetime import timezone

from qmod.config import AppConfig
from qmod.data_importer import _ensure_utc_naive_index

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class DatasetBuilder:
    """
    Thin wrapper around data loading for backtests.

    It accepts either:
      - a legacy CONFIG dict: {"ticker", "start_date", "end_date", "interval"}
      - an AppConfig instance (qmod.config.AppConfig)

    And exposes:
      - get_full_dataframe() -> OHLCV DataFrame
    """

    def __init__(self, config: Union[Dict[str, Any], AppConfig]) -> None:
        # Allow both dict-style legacy config and AppConfig DTO
        if isinstance(config, AppConfig):
            self.ticker = config.tkr
            self.start = config.start.isoformat() if config.start else None
            self.end = config.end.isoformat() if config.end else None
            self.interval = config.interval
        else:
            cfg = dict(config) if config is not None else {}
            self.ticker = cfg.get("ticker") or cfg.get("tkr")
            self.start = cfg.get("start") or cfg.get("start_date")
            self.end = cfg.get("end") or cfg.get("end_date")
            self.interval = cfg.get("interval", "1d")

        if not self.ticker:
            raise ValueError("DatasetBuilder requires a ticker in config ('ticker' or 'tkr').")

        logger.info(
            "DatasetBuilder init: ticker=%s, start=%s, end=%s, interval=%s",
            self.ticker,
            self.start,
            self.end,
            self.interval,
        )

    # -----------------------------------------------------

    def get_full_dataframe(self) -> pd.DataFrame:
        """
        Download full OHLCV history according to the config.
        Index is returned as UTC-naive DatetimeIndex via _ensure_utc_naive_index.
        """
        logger.info("Downloading OHLCV for %s", self.ticker)
        df = yf.download(
            self.ticker,
            start=self.start,
            end=self.end,
            interval=self.interval,
            auto_adjust=False,
            progress=False,
        )

        if df.empty:
            logger.warning("No data returned for %s", self.ticker)
            return df

        df = _ensure_utc_naive_index(df)
        return df
