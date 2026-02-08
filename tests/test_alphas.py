"""Tests for qlab.alphas."""

import numpy as np
import pandas as pd
import pytest

from qlab.alphas import (
    momentum,
    short_term_reversal,
    trend_strength,
    mean_reversion_zscore,
    rsi_signal,
    bollinger_signal,
    low_volatility,
    profitability_proxy,
    stability,
)


class TestMomentum:
    def test_momentum_shape(self, close_series):
        sig = momentum(close_series, lookback=40, skip=5)
        assert len(sig) == len(close_series)

    def test_momentum_skip_lt_lookback(self, close_series):
        with pytest.raises(ValueError, match="skip must be less"):
            momentum(close_series, lookback=10, skip=20)

    def test_reversal_is_negative_return(self, close_series):
        rev = short_term_reversal(close_series, lookback=10)
        # Reversal should be negative of 10-day return
        from qlab.features.returns import simple_returns
        ret10 = simple_returns(close_series, periods=10)
        common = rev.dropna().index.intersection(ret10.dropna().index)
        np.testing.assert_allclose(
            rev.loc[common].values, -ret10.loc[common].values, rtol=1e-10
        )

    def test_trend_strength_shape(self, close_series):
        sig = trend_strength(close_series, lookback=40)
        assert len(sig) == len(close_series)


class TestMeanReversion:
    def test_mr_zscore_shape(self, close_series):
        sig = mean_reversion_zscore(close_series, lookback=20)
        assert len(sig) == len(close_series)

    def test_rsi_bounded(self, close_series):
        sig = rsi_signal(close_series, period=14)
        valid = sig.dropna()
        assert valid.min() >= -1.0 - 1e-10
        assert valid.max() <= 1.0 + 1e-10

    def test_bollinger_shape(self, close_series):
        sig = bollinger_signal(close_series, window=20, num_std=2.0)
        assert len(sig) == len(close_series)


class TestLowVol:
    def test_low_vol_negative_values(self, close_series):
        sig = low_volatility(close_series, lookback=40)
        # Low vol signal = -vol, so should be non-positive
        valid = sig.dropna()
        assert (valid <= 0).all()


class TestQuality:
    def test_profitability_proxy_shape(self, close_series):
        sig = profitability_proxy(close_series, lookback=40)
        assert len(sig) == len(close_series)

    def test_stability_bounded(self, close_series):
        sig = stability(close_series, lookback=40)
        valid = sig.dropna()
        # R² should be in [0, 1]
        assert valid.min() >= -1e-10
        assert valid.max() <= 1.0 + 1e-10
