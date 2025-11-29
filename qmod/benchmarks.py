from __future__ import annotations

import re
from typing import Dict, Tuple

# Extensible suffix → benchmark map
SUFFIX_TO_BENCH: Dict[str, str] = {
    # Your rules
    "HK": "^HSI",
    "SS": "^DJSH",
    "NS": "^DJSH",
    # Add more as you standardize (e.g., "LN": "^FTSE", "SW": "^SSMI")
}

DEFAULT_BENCHMARK = "SPY"  # No suffix → SPY

def _extract_suffix(tkr: str) -> str:
    """
    Extract a likely market suffix given common patterns:
      - '0005.HK'  -> HK
      - '600519.SS'-> SS
      - 'TCS.NS'   -> NS
      - 'AAPL'     -> '' (no suffix)
      - 'ETH-USD'  -> USD (we'll map to default unless explicitly defined)
    """
    m = re.search(r"[.\-]([A-Za-z]{2,4})$", tkr)
    return m.group(1).upper() if m else ""

def pick_benchmark(tkr: str) -> Tuple[str, str]:
    """
    Returns (suffix, chosen_benchmark_ticker)
    """
    suffix = _extract_suffix(tkr)
    bench = SUFFIX_TO_BENCH.get(suffix, DEFAULT_BENCHMARK)
    return suffix, bench
