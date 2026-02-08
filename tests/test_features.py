"""Tests for qlab.features."""

import numpy as np
import pandas as pd
import pytest

from qlab.features import (
    simple_returns,
    log_returns,
    cumulative_returns,
    excess_returns,
    realized_volatility,
    ewm_volatility,
    parkinson_volatility,
    garman_klass_volatility,
    rolling_mean,
    rolling_std,
    rolling_skew,
    rolling_beta,
    rank,
    zscore,
    demean,
    winsorize,
    neutralize,
)


class TestReturns:
    def test_simple_returns_shape(self, close_series):
        ret = simple_returns(close_series)
        # Same length, first obs per ticker is NaN
        assert len(ret) == len(close_series)

    def test_simple_returns_values(self):
        idx = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2020-01-01"), "A"),
                (pd.Timestamp("2020-01-02"), "A"),
                (pd.Timestamp("2020-01-03"), "A"),
            ],
            names=["date", "ticker"],
        )
        prices = pd.Series([100.0, 110.0, 105.0], index=idx)
        ret = simple_returns(prices)
        np.testing.assert_allclose(ret.dropna().values, [0.1, -1 / 22], rtol=1e-10)

    def test_log_returns_close_to_simple_for_small_changes(self, close_series):
        sr = simple_returns(close_series).dropna()
        lr = log_returns(close_series).dropna()
        # For small returns, log ≈ simple
        np.testing.assert_allclose(sr.values, lr.values, atol=0.01)

    def test_excess_returns(self, close_series, sample_dates):
        ret = simple_returns(close_series).dropna()
        bench = pd.Series(0.001, index=sample_dates, name="bench")
        er = excess_returns(ret, bench)
        # excess should be ret - 0.001
        expected = ret.values - 0.001
        np.testing.assert_allclose(er.values, expected, atol=1e-12)


class TestVolatility:
    def test_realized_vol_positive(self, sample_returns):
        vol = realized_volatility(sample_returns, window=20)
        assert (vol.dropna() >= 0).all()

    def test_ewm_vol_positive(self, sample_returns):
        vol = ewm_volatility(sample_returns, halflife=20)
        assert (vol.dropna() >= 0).all()

    def test_parkinson_positive(self, sample_prices):
        vol = parkinson_volatility(
            sample_prices["high"], sample_prices["low"], window=20
        )
        assert (vol.dropna() >= 0).all()

    def test_garman_klass_positive(self, sample_prices):
        vol = garman_klass_volatility(
            sample_prices["open"],
            sample_prices["high"],
            sample_prices["low"],
            sample_prices["close"],
            window=20,
        )
        assert (vol.dropna() >= 0).all()


class TestRolling:
    def test_rolling_mean_constant(self):
        idx = pd.MultiIndex.from_arrays(
            [pd.bdate_range("2020-01-01", periods=10), ["A"] * 10],
            names=["date", "ticker"],
        )
        s = pd.Series(5.0, index=idx)
        result = rolling_mean(s, window=3)
        np.testing.assert_allclose(result.dropna().values, 5.0)

    def test_rolling_std_constant_is_zero(self):
        idx = pd.MultiIndex.from_arrays(
            [pd.bdate_range("2020-01-01", periods=10), ["A"] * 10],
            names=["date", "ticker"],
        )
        s = pd.Series(5.0, index=idx)
        result = rolling_std(s, window=3)
        np.testing.assert_allclose(result.dropna().values, 0.0, atol=1e-15)


class TestCrossSection:
    def test_rank_bounds(self, sample_signal):
        r = rank(sample_signal)
        assert r.dropna().min() > 0
        assert r.dropna().max() <= 1.0

    def test_zscore_mean_zero(self, sample_signal):
        z = zscore(sample_signal)
        means = z.groupby(level="date").mean()
        np.testing.assert_allclose(means.values, 0.0, atol=1e-10)

    def test_demean_mean_zero(self, sample_signal):
        d = demean(sample_signal)
        means = d.groupby(level="date").mean()
        np.testing.assert_allclose(means.values, 0.0, atol=1e-10)

    def test_winsorize_clips(self, sample_signal):
        w = winsorize(sample_signal, lower=0.1, upper=0.9)
        # Winsorized range should be <= raw range per date
        raw_range = sample_signal.groupby(level="date").apply(
            lambda g: g.max() - g.min()
        )
        win_range = w.groupby(level="date").apply(lambda g: g.max() - g.min())
        assert (win_range <= raw_range + 1e-10).all()

    def test_neutralize_within_group_mean(self, sample_signal, tickers):
        # Use ticker as group (trivial -- each group has 1 element)
        groups = pd.Series(
            sample_signal.index.get_level_values("ticker"),
            index=sample_signal.index,
        )
        n = neutralize(sample_signal, groups)
        # With each group being a single ticker, neutralized = demeaned within singleton = 0
        np.testing.assert_allclose(n.values, 0.0, atol=1e-10)
