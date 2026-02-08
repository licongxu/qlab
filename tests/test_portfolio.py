"""Tests for qlab.portfolio."""

import numpy as np
import pandas as pd
import pytest

from qlab.portfolio.construction import (
    equal_weight_long_short,
    quantile_weights,
    proportional_weights,
    normalize_weights,
)
from qlab.portfolio.constraints import (
    apply_position_limits,
    apply_turnover_limit,
    dollar_neutral,
)


class TestConstruction:
    def test_equal_weight_dollar_neutral(self, sample_signal):
        w = equal_weight_long_short(sample_signal, long_pct=0.4, short_pct=0.4)
        # Net exposure should be ~0 per date
        net = w.groupby(level="date").sum()
        np.testing.assert_allclose(net.values, 0.0, atol=1e-10)

    def test_equal_weight_has_longs_and_shorts(self, sample_signal):
        w = equal_weight_long_short(sample_signal, long_pct=0.4, short_pct=0.4)
        assert (w > 0).any()
        assert (w < 0).any()

    def test_quantile_weights_dollar_neutral(self, sample_signal):
        w = quantile_weights(sample_signal, n_quantiles=5, long_quantile=5, short_quantile=1)
        net = w.groupby(level="date").sum()
        np.testing.assert_allclose(net.values, 0.0, atol=1e-10)

    def test_proportional_long_only(self, sample_signal):
        w = proportional_weights(sample_signal, long_only=True)
        assert (w.dropna() >= -1e-10).all()

    def test_normalize_gross_exposure(self, sample_signal):
        w = proportional_weights(sample_signal)
        wn = normalize_weights(w, gross_exposure=2.0, net_exposure=0.0)
        gross = wn.abs().groupby(level="date").sum()
        np.testing.assert_allclose(gross.values, 2.0, atol=1e-6)


class TestConstraints:
    def test_position_limits(self, sample_weights):
        capped = apply_position_limits(sample_weights, max_weight=0.3, min_weight=-0.3)
        assert capped.max() <= 0.3 + 1e-6
        assert capped.min() >= -0.3 - 1e-6

    def test_dollar_neutral(self, sample_weights):
        dn = dollar_neutral(sample_weights)
        net = dn.groupby(level="date").sum()
        np.testing.assert_allclose(net.values, 0.0, atol=1e-10)

    def test_turnover_limit(self, sample_weights):
        new_w = sample_weights * 2  # exaggerated new weights
        limited = apply_turnover_limit(new_w, sample_weights, max_turnover=0.05)
        delta = (limited - sample_weights).abs().groupby(level="date").sum() / 2
        assert (delta <= 0.05 + 1e-8).all()
