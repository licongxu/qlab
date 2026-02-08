"""Tests for qlab.utils."""

import numpy as np
import pandas as pd
import pytest

from qlab.utils.validation import (
    validate_prices,
    validate_signal,
    validate_weights,
    QlabValidationError,
)
from qlab.utils.calendar import trading_days, rebalance_dates, is_month_end
from qlab.utils.alignment import stack_prices, unstack_to_wide, align_frames


class TestValidation:
    def test_validate_prices_good(self, sample_prices):
        validate_prices(sample_prices)  # should not raise

    def test_validate_prices_not_dataframe(self):
        with pytest.raises(QlabValidationError, match="must be a DataFrame"):
            validate_prices("not a df")

    def test_validate_prices_missing_columns(self, sample_prices):
        bad = sample_prices.drop(columns=["close"])
        with pytest.raises(QlabValidationError, match="missing required columns"):
            validate_prices(bad)

    def test_validate_prices_no_multiindex(self):
        df = pd.DataFrame({"open": [1], "high": [2], "low": [0.5], "close": [1.5], "volume": [100]})
        with pytest.raises(QlabValidationError, match="MultiIndex"):
            validate_prices(df)

    def test_validate_signal_good(self, sample_signal):
        validate_signal(sample_signal)

    def test_validate_signal_not_series(self):
        with pytest.raises(QlabValidationError, match="must be a Series"):
            validate_signal(42)

    def test_validate_weights_good(self, sample_weights):
        validate_weights(sample_weights)


class TestCalendar:
    def test_trading_days_excludes_weekends(self):
        days = trading_days("2020-01-01", "2020-01-10")
        # No Saturdays or Sundays
        assert all(d.weekday() < 5 for d in days)

    def test_rebalance_dates_daily_returns_all(self):
        days = trading_days("2020-01-01", "2020-03-31")
        reb = rebalance_dates(days, freq="daily")
        assert len(reb) == len(days)

    def test_rebalance_dates_monthly_less_than_daily(self):
        days = trading_days("2020-01-01", "2020-12-31")
        reb = rebalance_dates(days, freq="monthly")
        assert len(reb) < len(days)
        assert len(reb) >= 11  # at least 11 month-ends

    def test_is_month_end_count(self):
        days = trading_days("2020-01-01", "2020-12-31")
        me = is_month_end(days)
        assert me.sum() == 12  # 12 month-ends in a year


class TestAlignment:
    def test_stack_unstack_roundtrip(self):
        dates = pd.bdate_range("2020-01-01", periods=5)
        wide = pd.DataFrame(
            {"A": range(5), "B": range(5, 10)},
            index=dates,
        ).astype(float)
        stacked = stack_prices(wide)
        unstacked = unstack_to_wide(stacked, column="close")
        unstacked.index.name = None
        unstacked.columns.name = None
        pd.testing.assert_frame_equal(unstacked.sort_index(axis=1), wide.sort_index(axis=1))

    def test_align_frames_inner(self):
        idx1 = pd.RangeIndex(5)
        idx2 = pd.RangeIndex(3, 8)
        s1 = pd.Series(range(5), index=idx1)
        s2 = pd.Series(range(5), index=idx2)
        a1, a2 = align_frames(s1, s2, how="inner")
        assert len(a1) == 2  # indices 3, 4
        assert len(a2) == 2
