"""Regression tests for ticker parsing, normalization, and validation."""

from __future__ import annotations

import pytest

from qlab.data.ticker_utils import parse_tickers, normalize_ticker, validate_tickers
from qlab.utils.validation import QlabValidationError


class TestParseTickers:
    """Test all accepted input formats."""

    def test_single_ticker(self):
        assert parse_tickers("NVDA") == ["NVDA"]

    def test_comma_separated(self):
        assert parse_tickers("AAPL,MSFT,NVDA") == ["AAPL", "MSFT", "NVDA"]

    def test_comma_space_separated(self):
        assert parse_tickers("AAPL, MSFT, NVDA") == ["AAPL", "MSFT", "NVDA"]

    def test_space_separated(self):
        assert parse_tickers("AAPL MSFT NVDA") == ["AAPL", "MSFT", "NVDA"]

    def test_stringified_python_list(self):
        assert parse_tickers("['AAPL','MSFT']") == ["AAPL", "MSFT"]

    def test_json_array_string(self):
        assert parse_tickers('["AAPL","MSFT"]') == ["AAPL", "MSFT"]

    def test_actual_list(self):
        assert parse_tickers(["AAPL", "MSFT"]) == ["AAPL", "MSFT"]

    def test_lowercase_normalized(self):
        assert parse_tickers("aapl,msft") == ["AAPL", "MSFT"]

    def test_whitespace_stripped(self):
        assert parse_tickers("  AAPL , MSFT  ") == ["AAPL", "MSFT"]

    def test_duplicates_removed(self):
        assert parse_tickers("AAPL,AAPL,MSFT") == ["AAPL", "MSFT"]

    def test_empty_entries_dropped(self):
        assert parse_tickers("AAPL,,MSFT,") == ["AAPL", "MSFT"]

    def test_empty_string(self):
        assert parse_tickers("") == []

    def test_empty_list(self):
        assert parse_tickers([]) == []

    def test_mixed_case_list(self):
        assert parse_tickers(["aapl", "MSFT", " nvda "]) == ["AAPL", "MSFT", "NVDA"]

    def test_ticker_with_dot(self):
        assert parse_tickers("BRK.B") == ["BRK.B"]

    def test_ticker_with_hyphen(self):
        assert parse_tickers("BF-B") == ["BF-B"]

    def test_invalid_characters_dropped(self):
        # Tickers with special chars like $ or @  are dropped
        result = parse_tickers("AAPL,$BAD,MSFT")
        assert result == ["AAPL", "MSFT"]

    def test_stringified_list_with_spaces(self):
        assert parse_tickers("['AAPL', 'MSFT', 'GOOG']") == ["AAPL", "MSFT", "GOOG"]


class TestNormalizeTicker:
    def test_basic(self):
        assert normalize_ticker("aapl") == "AAPL"

    def test_strip_whitespace(self):
        assert normalize_ticker("  MSFT  ") == "MSFT"

    def test_strip_quotes(self):
        assert normalize_ticker("'AAPL'") == "AAPL"
        assert normalize_ticker('"AAPL"') == "AAPL"

    def test_invalid_chars(self):
        assert normalize_ticker("$BAD") == ""

    def test_empty(self):
        assert normalize_ticker("") == ""
        assert normalize_ticker("   ") == ""


class TestValidateTickers:
    def test_all_valid(self):
        valid, invalid = validate_tickers(["AAPL", "MSFT"])
        assert valid == ["AAPL", "MSFT"]
        assert invalid == []

    def test_some_invalid(self):
        valid, invalid = validate_tickers(["AAPL", "$BAD", "MSFT"])
        assert valid == ["AAPL", "MSFT"]
        assert invalid == ["$BAD"]

    def test_none_valid_raises(self):
        with pytest.raises(QlabValidationError):
            validate_tickers(["$BAD", "@WORSE"])
