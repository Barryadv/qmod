from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


class BaseDTO(BaseModel):
    """Base DTO with JSON helper."""
    model_config = ConfigDict(from_attributes=True)

    def to_json(
        self,
        *,
        indent: int | None = None,
        by_alias: bool = False,
        exclude_none: bool = False,
    ) -> str:
        """
        Serialize the DTO to a JSON string using Pydantic's model_dump_json.
        Datetime fields are rendered in ISO-8601 format by default.
        """
        return self.model_dump_json(
            indent=indent,
            by_alias=by_alias,
            exclude_none=exclude_none,
        )


class PriceBarDTO(BaseDTO):
    """Single OHLCV bar."""
    t: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None

    @model_validator(mode="after")
    def _validate_ranges(self) -> "PriceBarDTO":
        if self.high < self.low:
            raise ValueError("high must be >= low")
        if not (self.low <= self.open <= self.high):
            raise ValueError("open must be within [low, high]")
        if not (self.low <= self.close <= self.high):
            raise ValueError("close must be within [low, high]")
        if self.volume is not None and self.volume < 0:
            raise ValueError("volume must be >= 0 when provided")
        return self


class IndicatorDTO(BaseDTO):
    """Indicator single-point snapshot."""
    name: str
    params: dict[str, Any]
    values: dict[str, float]

    @field_validator("name")
    @classmethod
    def _name_non_empty(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("name cannot be empty")
        return v2

    @field_validator("values")
    @classmethod
    def _values_non_empty(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("values cannot be empty")
        return v


class SignalDTO(BaseDTO):
    """Trading signal."""
    date: datetime
    ticker: str
    signal: Literal["buy", "sell", "hold"]
    score: float
    reasons: list[str] = Field(default_factory=list)

    @field_validator("ticker")
    @classmethod
    def _ticker_non_empty(cls, v: str) -> str:
        v2 = v.strip().upper()
        if not v2:
            raise ValueError("ticker cannot be empty")
        return v2

    @field_validator("reasons")
    @classmethod
    def _sanitize_reasons(cls, v: list[str]) -> list[str]:
        return [r.strip() for r in v if isinstance(r, str) and r.strip()]


class ChartDTO(BaseDTO):
    """Chart payload as base64-encoded image."""
    format: Literal["png_base64"]
    data: str

    @field_validator("data")
    @classmethod
    def _data_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("data cannot be empty")
        return v.strip()


class LLMRequestDTO(BaseDTO):
    """Request to an LLM for decision/explanation generation."""
    ticker: str
    date: datetime
    indicators: list[IndicatorDTO] = Field(default_factory=list)
    last_signal: SignalDTO | None = None
    prompt_hint: str | None = None
    chart: ChartDTO | None = None

    @field_validator("ticker")
    @classmethod
    def _ticker_non_empty(cls, v: str) -> str:
        v2 = v.strip().upper()
        if not v2:
            raise ValueError("ticker cannot be empty")
        return v2

    @field_validator("prompt_hint")
    @classmethod
    def _hint_trim(cls, v: str | None) -> str | None:
        return v.strip() if isinstance(v, str) else None


class LLMResponseDTO(BaseDTO):
    """Response from an LLM with signal and rationale."""
    signal: Literal["buy", "sell", "hold"]
    explanation: str
    chart_base64: str | None = None
    model_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("explanation")
    @classmethod
    def _explanation_trim(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("explanation cannot be empty")
        return v2

    @field_validator("model_version")
    @classmethod
    def _model_version_trim(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("model_version cannot be empty")
        return v2


class RunOutputDTO(BaseDTO):
    """
    Final JSON payload combining input, model outputs, and sanitized config.
    """
    ticker: str
    date: datetime
    signal: Literal["buy", "sell", "hold"]
    score: float
    explanation: str
    chart_base64: str | None = None
    model_version: str
    indicators: list[IndicatorDTO] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ticker")
    @classmethod
    def _ticker_non_empty(cls, v: str) -> str:
        v2 = v.strip().upper()
        if not v2:
            raise ValueError("ticker cannot be empty")
        return v2

    @field_validator("explanation")
    @classmethod
    def _explanation_trim(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("explanation cannot be empty")
        return v2

    @field_validator("model_version")
    @classmethod
    def _model_version_trim(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("model_version cannot be empty")
        return v2


# ===== Error DTO for API responses =====

class ErrorDTO(BaseDTO):
    """Structured error response for API."""
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


# ===== Composite report DTOs (bullets + chart) =====

class CompositeBulletDTO(BaseDTO):
    title: str
    body: str


class CompositeReportDTO(BaseDTO):
    # Inputs
    ticker: str
    start_date: datetime
    end_date: datetime
    opt_start_date: datetime | None = None  # optimization window (if used)
    opt_end_date: datetime | None = None

    # Indicator params & point-in-time values (last bar)
    macd_params: dict[str, int] = Field(default_factory=dict)  # {"fast":12, "slow":26, "signal":9}
    rsi_params: dict[str, int] = Field(default_factory=dict)   # {"length":14, "ob":70, "os":30}
    indicators: list[IndicatorDTO] = Field(default_factory=list)

    # UX/Layout hint and content
    layout: Literal["two_column", "stacked"] = "two_column"
    bullets: list[CompositeBulletDTO] = Field(default_factory=list)
    chart: ChartDTO

    # Optional signal snapshot
    last_signal: SignalDTO | None = None


__all__ = [
    "BaseDTO",
    "PriceBarDTO",
    "IndicatorDTO",
    "SignalDTO",
    "ChartDTO",
    "LLMRequestDTO",
    "LLMResponseDTO",
    "RunOutputDTO",
    "ErrorDTO",
    "CompositeBulletDTO",
    "CompositeReportDTO",
]
