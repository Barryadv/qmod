from __future__ import annotations

import logging
import os
import json
from datetime import date
from typing import Any, Dict, Optional, Literal

import numpy as np
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

# Prefer Pydantic Settings (v2) if available; fall back to BaseModel for portability.
try:
    from pydantic_settings import BaseSettings  # type: ignore
except Exception:  # pragma: no cover
    class BaseSettings(BaseModel):  # type: ignore
        """Fallback when pydantic-settings is not installed."""
        pass


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# ---------- Legacy-style global CONFIG for backtests ----------

CONFIG: Dict[str, Any] | None = None


def init_CONFIG(path: str | None = None) -> None:
    """
    Initializes global CONFIG from a JSON file or default values.
    If no file provided, creates a minimal config using environment variables or defaults.
    """
    global CONFIG

    # Case 1: load from JSON file if provided
    if path is not None and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                CONFIG = json.load(f)
                logger.info("Loaded CONFIG from %s", path)
                return
        except Exception as exc:
            logger.warning("Failed to load CONFIG from %s (%s); falling back to defaults", path, exc)

    # Case 2: fallback default config (minimal but valid)
    ticker = os.getenv("TKR", "AAPL")
    CONFIG = {
        "ticker": ticker,
        "start_date": "2019-01-01",
        "end_date": None,
        "interval": "1d",
    }
    logger.info("Initialized default CONFIG: %s", CONFIG)


def get_config() -> Dict[str, Any]:
    """Safe access to legacy CONFIG."""
    global CONFIG
    if CONFIG is None:
        init_CONFIG()
    return CONFIG


# ---------- Technical / AppConfig DTOs ----------

class TechnicalParams(BaseModel):
    sma_windows: list[int] = Field(default_factory=lambda: [20, 50])
    ema_windows: list[int] = Field(default_factory=lambda: [20])
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_std: float = 2.0

    @field_validator("sma_windows", "ema_windows")
    @classmethod
    def _validate_windows(cls, v: list[int]) -> list[int]:
        if any(w <= 0 for w in v):
            raise ValueError("Indicator windows must be positive integers.")
        return sorted(set(v))

    @field_validator("rsi_window", "macd_fast", "macd_slow", "macd_signal", "bb_window")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Window parameters must be positive.")
        return v

    @field_validator("bb_std")
    @classmethod
    def _bb_std_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Bollinger std must be positive.")
        return v


class AppConfig(BaseSettings):
    # fields (make sure these match)
    tkr: str = "AAPL"
    interval: Literal["1d"] = "1d"
    start: date | None = None
    end: date | None = None
    ti_params: TechnicalParams = Field(default_factory=TechnicalParams)
    db_url: str = "sqlite:///quant.db"
    store_charts_in_db: bool = True
    store_charts_in_json: bool = True
    model_version: str = "v0.1.0"

    @model_validator(mode="after")
    def _validate_dates(self) -> "AppConfig":
        if self.start and self.end and self.start > self.end:
            raise ValueError("start must be <= end")
        return self

    @classmethod
    def from_env_or_defaults(cls, **overrides: Any) -> "AppConfig":
        """
        Load from env/defaults, then apply overrides and re-validate so that
        'start'/'end' strings like '2024-01-01' become `date` objects.
        """
        try:
            np.random.seed(42)  # determinism convention
        except Exception:
            pass

        # 1) Base from env/defaults (parses types)
        base = cls()

        # 2) Merge overrides and re-validate to coerce types (e.g., str -> date)
        if overrides:
            merged = base.model_dump()
            merged.update(overrides)
            cfg = cls.model_validate(merged)
        else:
            cfg = base

        # 3) Optional env override for ticker if not explicitly overridden
        tkr_env = os.getenv("TKR")
        if tkr_env and "tkr" not in overrides:
            cfg = cfg.model_copy(update={"tkr": tkr_env})

        return cfg


__all__ = [
    "TechnicalParams",
    "AppConfig",
    "CONFIG",
    "init_CONFIG",
    "get_config",
]
