"""Ticker parsing, normalization, and validation utilities.

Handles common input formats from CLI, API, and GUI entry points and
normalises them into a consistent canonical form suitable for data providers
and cache key generation.
"""

from __future__ import annotations

import json
import re
import warnings
from typing import Sequence

from qlab.utils.validation import QlabValidationError

# Valid ticker characters: A-Z, 0-9, dot, hyphen
_TICKER_RE = re.compile(r"^[A-Z0-9.\-]+$")


def parse_tickers(raw: str | list | Sequence[str]) -> list[str]:
    """Parse ticker input from various formats into a clean list.

    Accepted formats::

        "NVDA"                           # single ticker
        "AAPL,MSFT,NVDA"                # comma-separated
        "AAPL MSFT NVDA"                # space-separated
        "AAPL, MSFT, NVDA"              # comma+space
        "['AAPL','MSFT']"               # stringified Python list
        '["AAPL","MSFT"]'               # JSON array string
        ["AAPL", "MSFT"]                # actual list/sequence

    Returns a deduplicated list of normalized tickers (empty/invalid entries
    are silently dropped).
    """
    if isinstance(raw, (list, tuple)):
        tokens = [str(t) for t in raw]
    elif isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        # Try JSON array first (also handles stringified Python lists with double quotes)
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    tokens = [str(t) for t in parsed]
                else:
                    tokens = [s]
            except json.JSONDecodeError:
                # Handle Python-style list: "['AAPL','MSFT']"
                cleaned = s.strip("[]")
                cleaned = cleaned.replace("'", "").replace('"', "")
                tokens = re.split(r"[,\s]+", cleaned)
        else:
            # Comma and/or space separated
            tokens = re.split(r"[,\s]+", s)
    else:
        tokens = [str(raw)]

    result: list[str] = []
    seen: set[str] = set()
    for t in tokens:
        normed = normalize_ticker(t)
        if normed and normed not in seen:
            result.append(normed)
            seen.add(normed)
    return result


def normalize_ticker(ticker: str) -> str:
    """Normalize a single ticker string.

    Strips whitespace, removes surrounding quotes, converts to uppercase.
    Returns an empty string if the result is not a valid ticker.
    """
    t = ticker.strip().strip("'\"").strip().upper()
    if not t:
        return ""
    if not _TICKER_RE.match(t):
        return ""
    return t


def validate_tickers(tickers: list[str]) -> tuple[list[str], list[str]]:
    """Validate a list of tickers.

    Returns ``(valid, invalid)`` where *valid* contains tickers matching
    the allowed character set and *invalid* contains the rest.

    Raises :class:`QlabValidationError` if no valid tickers remain.
    """
    valid: list[str] = []
    invalid: list[str] = []
    for t in tickers:
        if _TICKER_RE.match(t):
            valid.append(t)
        else:
            invalid.append(t)
    if invalid:
        warnings.warn(f"Dropped invalid tickers: {invalid}")
    if not valid:
        raise QlabValidationError(
            f"No valid tickers after validation. Input: {tickers}"
        )
    return valid, invalid
