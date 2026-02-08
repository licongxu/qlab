"""Unit tests for the StockPicker pipeline.

Tests cover:
  - Data alignment (MultiIndex consistency, no NaN leaks)
  - Signal calculation consistency (deterministic, correct shapes)
  - Selection constraints (sector cap, max position weight)
  - No-lookahead checks (signals use only past data)
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.config import SelectionConfig
from app.stock_picker import StockPicker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_output_dir(tmp_path):
    """Temporary output directory for test runs."""
    d = tmp_path / "test_runs"
    d.mkdir()
    return str(d)


@pytest.fixture()
def universe_csv(tmp_path) -> str:
    """Small test universe CSV with 10 tickers across 4 sectors."""
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "JPM", "BAC", "JNJ", "PFE", "XOM"]
    sectors = [
        "Information Technology", "Information Technology", "Communication Services",
        "Consumer Discretionary", "Communication Services",
        "Financials", "Financials", "Health Care", "Health Care", "Energy",
    ]
    df = pd.DataFrame({"ticker": tickers, "sector": sectors})
    path = tmp_path / "test_universe.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def small_config(universe_csv, tmp_output_dir) -> SelectionConfig:
    """Config for fast test runs with real data."""
    return SelectionConfig(
        universe_file=universe_csv,
        start_date="2024-01-01",
        end_date="2025-01-01",
        n_long=5,
        long_only=True,
        n_short=0,
        max_position_weight=0.25,
        sector_cap=0.50,
        min_dollar_volume_20d=1e6,
        max_volatility_annualized=1.5,
        min_price=1.0,
        regime_filter=True,
        output_dir=tmp_output_dir,
    )


@pytest.fixture()
def synthetic_close() -> pd.Series:
    """Synthetic close price series for 10 tickers, 300 days, deterministic."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-02", periods=300, freq="B")
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "JPM", "BAC", "JNJ", "PFE", "XOM"]
    parts = []
    for t in tickers:
        n = len(dates)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
        s = pd.Series(close, index=pd.MultiIndex.from_arrays(
            [dates, np.full(n, t)], names=["date", "ticker"]
        ), name="adj_close")
        parts.append(s)
    return pd.concat(parts).sort_index()


@pytest.fixture()
def synthetic_volume(synthetic_close) -> pd.Series:
    """Synthetic volume series matching synthetic_close."""
    rng = np.random.default_rng(99)
    vals = rng.integers(1_000_000, 10_000_000, len(synthetic_close)).astype(float)
    return pd.Series(vals, index=synthetic_close.index, name="volume")


@pytest.fixture()
def synthetic_prices(synthetic_close, synthetic_volume) -> pd.DataFrame:
    """Full OHLCV DataFrame from synthetic data."""
    rng = np.random.default_rng(7)
    n = len(synthetic_close)
    c = synthetic_close.values
    return pd.DataFrame({
        "open": c * np.exp(rng.normal(0, 0.003, n)),
        "high": c * (1 + rng.uniform(0, 0.02, n)),
        "low": c * (1 - rng.uniform(0, 0.02, n)),
        "close": c,
        "adj_close": c,
        "volume": synthetic_volume.values,
    }, index=synthetic_close.index)


# ---------------------------------------------------------------------------
# 1. Data alignment tests
# ---------------------------------------------------------------------------

class TestDataAlignment:
    """Verify that MultiIndex structure is preserved through pipeline stages."""

    def test_close_has_multiindex(self, synthetic_close):
        assert isinstance(synthetic_close.index, pd.MultiIndex)
        assert synthetic_close.index.names == ["date", "ticker"]

    def test_all_tickers_present(self, synthetic_close):
        tickers = synthetic_close.index.get_level_values("ticker").unique()
        assert len(tickers) == 10

    def test_filter_preserves_multiindex(self, synthetic_close, synthetic_volume, synthetic_prices):
        config = SelectionConfig(min_price=0.01, min_dollar_volume_20d=100, max_volatility_annualized=5.0)
        picker = StockPicker(config)
        filtered, log = picker._apply_filters(synthetic_close, synthetic_volume, synthetic_prices)
        assert isinstance(filtered, list)
        assert all(isinstance(t, str) for t in filtered)
        # With lenient filters, should keep all tickers
        assert len(filtered) == 10

    def test_filter_removes_low_price(self, synthetic_close, synthetic_volume, synthetic_prices):
        """Stocks below min_price should be filtered out."""
        # Modify one ticker to have very low prices
        modified = synthetic_close.copy()
        mask = modified.index.get_level_values("ticker") == "XOM"
        modified.loc[mask] = 0.50  # Below default min_price of 5.0
        config = SelectionConfig(min_price=5.0, min_dollar_volume_20d=100, max_volatility_annualized=5.0)
        picker = StockPicker(config)
        filtered, log = picker._apply_filters(modified, synthetic_volume, synthetic_prices)
        assert "XOM" not in filtered
        filter_reasons = {l["ticker"]: l["filter"] for l in log}
        assert filter_reasons.get("XOM") == "min_price"

    def test_signals_have_multiindex(self, synthetic_close):
        config = SelectionConfig()
        picker = StockPicker(config)
        tickers = synthetic_close.index.get_level_values("ticker").unique().tolist()
        signals, details = picker._compute_signals(synthetic_close, tickers)
        for name, sig in signals.items():
            assert isinstance(sig.index, pd.MultiIndex), f"{name} lost MultiIndex"
            assert sig.index.names == ["date", "ticker"]


# ---------------------------------------------------------------------------
# 2. Signal calculation consistency tests
# ---------------------------------------------------------------------------

class TestSignalConsistency:
    """Signals should be deterministic, properly normalized, finite."""

    def test_signals_deterministic(self, synthetic_close):
        config = SelectionConfig()
        picker = StockPicker(config)
        tickers = synthetic_close.index.get_level_values("ticker").unique().tolist()

        signals_1, _ = picker._compute_signals(synthetic_close, tickers)
        signals_2, _ = picker._compute_signals(synthetic_close, tickers)

        for name in signals_1:
            pd.testing.assert_series_equal(signals_1[name], signals_2[name])

    def test_signals_finite(self, synthetic_close):
        config = SelectionConfig()
        picker = StockPicker(config)
        tickers = synthetic_close.index.get_level_values("ticker").unique().tolist()
        signals, _ = picker._compute_signals(synthetic_close, tickers)
        for name, sig in signals.items():
            assert sig.isna().sum() == 0, f"{name} has NaN values after dropna"
            assert np.isfinite(sig.values).all(), f"{name} has non-finite values"

    def test_zscore_mean_near_zero(self, synthetic_close):
        """After z-score normalization, cross-sectional mean should be ~0."""
        config = SelectionConfig()
        picker = StockPicker(config)
        tickers = synthetic_close.index.get_level_values("ticker").unique().tolist()
        signals, _ = picker._compute_signals(synthetic_close, tickers)

        for name, sig in signals.items():
            cs_mean = sig.groupby(level="date").mean()
            assert abs(cs_mean.mean()) < 0.5, f"{name} cross-sectional mean too far from 0: {cs_mean.mean():.4f}"

    def test_composite_combines_all_signals(self, synthetic_close):
        config = SelectionConfig()
        picker = StockPicker(config)
        tickers = synthetic_close.index.get_level_values("ticker").unique().tolist()
        signals, _ = picker._compute_signals(synthetic_close, tickers)
        composite = picker._combine_signals(signals)

        assert isinstance(composite, pd.Series)
        assert isinstance(composite.index, pd.MultiIndex)
        assert len(composite) > 0

    def test_composite_zscore_normalized(self, synthetic_close):
        config = SelectionConfig()
        picker = StockPicker(config)
        tickers = synthetic_close.index.get_level_values("ticker").unique().tolist()
        signals, _ = picker._compute_signals(synthetic_close, tickers)
        composite = picker._combine_signals(signals)

        # Cross-sectional std should be close to 1
        cs_std = composite.groupby(level="date").std()
        mean_std = cs_std.mean()
        assert 0.5 < mean_std < 2.0, f"Composite std not normalized: {mean_std:.4f}"


# ---------------------------------------------------------------------------
# 3. Selection constraints tests
# ---------------------------------------------------------------------------

class TestSelectionConstraints:
    """Verify sector caps, position weight limits, and selection count."""

    def test_selection_count(self, synthetic_close, synthetic_volume, synthetic_prices):
        config = SelectionConfig(
            n_long=5, long_only=True, sector_cap=1.0,
            min_price=0.01, min_dollar_volume_20d=100, max_volatility_annualized=5.0,
        )
        picker = StockPicker(config)
        tickers = synthetic_close.index.get_level_values("ticker").unique().tolist()
        signals, _ = picker._compute_signals(synthetic_close, tickers)
        composite = picker._combine_signals(signals)
        regime_info = {"regime": "NORMAL", "current_vol": 0.15, "threshold": 0.25, "exposure_adjustment": 1.0}
        # Unique sector per ticker to avoid sector cap interference
        sector_map = {t: f"Sector_{i}" for i, t in enumerate(tickers)}

        selection = picker._generate_selection(composite, sector_map, regime_info)
        # Should select up to n_long (may be less if fewer tickers in composite on latest date)
        assert len(selection) <= 5
        assert len(selection) > 0

    def test_sector_cap_enforced(self, synthetic_close):
        """With sector_cap=0.30 and n_long=10, max 3 stocks from one sector."""
        config = SelectionConfig(
            n_long=10, sector_cap=0.30,
            min_price=0.01, min_dollar_volume_20d=100, max_volatility_annualized=5.0,
        )
        picker = StockPicker(config)
        tickers = synthetic_close.index.get_level_values("ticker").unique().tolist()
        signals, _ = picker._compute_signals(synthetic_close, tickers)
        composite = picker._combine_signals(signals)

        # Put all tickers in same sector to test cap
        sector_map = {t: "Tech" for t in tickers}
        regime_info = {"regime": "NORMAL", "current_vol": 0.15, "threshold": 0.25, "exposure_adjustment": 1.0}
        selection = picker._generate_selection(composite, sector_map, regime_info)

        max_per_sector = max(1, int(10 * 0.30))  # = 3
        sector_counts = selection["sector"].value_counts()
        for sector, count in sector_counts.items():
            assert count <= max_per_sector, f"Sector {sector} has {count} > {max_per_sector}"

    def test_max_position_weight(self, synthetic_close):
        config = SelectionConfig(
            n_long=5, max_position_weight=0.25,
            min_price=0.01, min_dollar_volume_20d=100, max_volatility_annualized=5.0,
        )
        picker = StockPicker(config)
        tickers = synthetic_close.index.get_level_values("ticker").unique().tolist()
        signals, _ = picker._compute_signals(synthetic_close, tickers)
        composite = picker._combine_signals(signals)
        sector_map = {t: f"Sector_{i}" for i, t in enumerate(tickers)}
        regime_info = {"regime": "NORMAL", "current_vol": 0.15, "threshold": 0.25, "exposure_adjustment": 1.0}

        selection = picker._generate_selection(composite, sector_map, regime_info)
        for _, row in selection.iterrows():
            assert row["weight"] <= config.max_position_weight + 1e-9, \
                f"{row['ticker']} weight {row['weight']} > max {config.max_position_weight}"

    def test_regime_reduces_exposure(self, synthetic_close):
        """In HIGH_VOL regime, weights should be reduced by exposure_adjustment."""
        config = SelectionConfig(
            n_long=5, max_position_weight=0.25,
            min_price=0.01, min_dollar_volume_20d=100, max_volatility_annualized=5.0,
        )
        picker = StockPicker(config)
        tickers = synthetic_close.index.get_level_values("ticker").unique().tolist()
        signals, _ = picker._compute_signals(synthetic_close, tickers)
        composite = picker._combine_signals(signals)
        sector_map = {t: f"Sector_{i}" for i, t in enumerate(tickers)}

        normal = {"regime": "NORMAL", "current_vol": 0.15, "threshold": 0.25, "exposure_adjustment": 1.0}
        high_vol = {"regime": "HIGH_VOL", "current_vol": 0.35, "threshold": 0.25, "exposure_adjustment": 0.7}

        sel_normal = picker._generate_selection(composite, sector_map, normal)
        sel_highvol = picker._generate_selection(composite, sector_map, high_vol)

        total_normal = sel_normal["weight"].sum()
        total_highvol = sel_highvol["weight"].sum()
        assert total_highvol < total_normal, \
            f"HIGH_VOL total weight {total_highvol} not < NORMAL {total_normal}"

    def test_selection_has_required_columns(self, synthetic_close):
        config = SelectionConfig(
            n_long=5,
            min_price=0.01, min_dollar_volume_20d=100, max_volatility_annualized=5.0,
        )
        picker = StockPicker(config)
        tickers = synthetic_close.index.get_level_values("ticker").unique().tolist()
        signals, _ = picker._compute_signals(synthetic_close, tickers)
        composite = picker._combine_signals(signals)
        sector_map = {t: "Tech" for t in tickers}
        regime_info = {"regime": "NORMAL", "current_vol": 0.15, "threshold": 0.25, "exposure_adjustment": 1.0}

        selection = picker._generate_selection(composite, sector_map, regime_info)
        required = {"ticker", "score", "rank", "weight", "sector", "reason_codes", "date"}
        assert required.issubset(set(selection.columns)), \
            f"Missing columns: {required - set(selection.columns)}"


# ---------------------------------------------------------------------------
# 4. No-lookahead tests
# ---------------------------------------------------------------------------

class TestNoLookahead:
    """Verify that signals at date t only use data up to date t."""

    def test_signal_at_date_uses_only_past(self, synthetic_close):
        """Changing future data should not affect signal on a past date."""
        config = SelectionConfig()
        picker = StockPicker(config)
        tickers = synthetic_close.index.get_level_values("ticker").unique().tolist()
        dates = synthetic_close.index.get_level_values("date").unique()

        # Compute signals on full data
        signals_full, _ = picker._compute_signals(synthetic_close, tickers)

        # Compute signals on truncated data (drop last 50 days)
        cutoff = dates[-50]
        truncated = synthetic_close[synthetic_close.index.get_level_values("date") < cutoff]
        signals_trunc, _ = picker._compute_signals(truncated, tickers)

        # For an early date, signals should be identical
        check_date = dates[150]  # Well before cutoff
        for name in signals_full:
            if name not in signals_trunc:
                continue
            full_at_date = signals_full[name].xs(check_date, level="date") if check_date in signals_full[name].index.get_level_values("date") else None
            trunc_at_date = signals_trunc[name].xs(check_date, level="date") if check_date in signals_trunc[name].index.get_level_values("date") else None

            if full_at_date is not None and trunc_at_date is not None:
                pd.testing.assert_series_equal(
                    full_at_date, trunc_at_date,
                    check_names=False, atol=1e-6,
                    obj=f"Signal {name} at {check_date}",
                )

    def test_selection_date_is_latest(self, synthetic_close):
        """Selection should be for the latest available date."""
        config = SelectionConfig(
            n_long=5,
            min_price=0.01, min_dollar_volume_20d=100, max_volatility_annualized=5.0,
        )
        picker = StockPicker(config)
        tickers = synthetic_close.index.get_level_values("ticker").unique().tolist()
        signals, _ = picker._compute_signals(synthetic_close, tickers)
        composite = picker._combine_signals(signals)
        sector_map = {t: "Tech" for t in tickers}
        regime_info = {"regime": "NORMAL", "current_vol": 0.15, "threshold": 0.25, "exposure_adjustment": 1.0}

        selection = picker._generate_selection(composite, sector_map, regime_info)
        latest_date = composite.index.get_level_values("date").max()
        assert all(selection["date"] == str(latest_date.date()))


# ---------------------------------------------------------------------------
# 5. Config validation tests
# ---------------------------------------------------------------------------

class TestConfig:
    """Verify SelectionConfig defaults and serialization."""

    def test_default_alpha_weights_sum_to_one(self):
        config = SelectionConfig()
        total = sum(config.alpha_weights.values())
        assert abs(total - 1.0) < 1e-9, f"Alpha weights sum to {total}, expected 1.0"

    def test_to_dict_roundtrip(self):
        config = SelectionConfig(n_long=30, sector_cap=0.25)
        d = config.to_dict()
        assert d["n_long"] == 30
        assert d["sector_cap"] == 0.25
        assert isinstance(d["alpha_weights"], dict)

    def test_config_frozen(self):
        config = SelectionConfig()
        with pytest.raises(AttributeError):
            config.n_long = 99


# ---------------------------------------------------------------------------
# 6. Output file tests
# ---------------------------------------------------------------------------

class TestOutputFiles:
    """Verify that the pipeline produces expected output files."""

    def test_run_dir_structure(self, synthetic_close, synthetic_volume, synthetic_prices, tmp_path):
        """Run with synthetic data and check output file structure."""
        # Create universe CSV
        tickers = synthetic_close.index.get_level_values("ticker").unique().tolist()
        sectors = [f"Sector_{i % 4}" for i in range(len(tickers))]
        universe = pd.DataFrame({"ticker": tickers, "sector": sectors})
        uni_path = tmp_path / "universe.csv"
        universe.to_csv(uni_path, index=False)

        config = SelectionConfig(
            universe_file=str(uni_path),
            start_date="2024-01-02",
            end_date="2025-04-01",
            n_long=5,
            output_dir=str(tmp_path / "runs"),
            min_price=0.01,
            min_dollar_volume_20d=100,
            max_volatility_annualized=5.0,
        )

        # Monkey-patch _fetch_data to return our synthetic data
        picker = StockPicker(config)

        def mock_fetch(tickers_arg):
            return synthetic_prices, synthetic_close, synthetic_volume

        picker._fetch_data = mock_fetch
        result = picker.run()

        run_dir = Path(result["output_dir"])
        assert (run_dir / "config.json").exists()
        assert (run_dir / "selection.csv").exists()
        assert (run_dir / "backtest_summary.json").exists()
        assert (run_dir / "filter_log.json").exists()
        assert (run_dir / "regime.json").exists()
        assert (run_dir / "signal_details.csv").exists()

        # Verify JSON is valid
        config_data = json.loads((run_dir / "config.json").read_text())
        assert config_data["n_long"] == 5

        # Verify selection CSV has correct number of stocks
        sel = pd.read_csv(run_dir / "selection.csv")
        assert len(sel) <= 5
        assert "ticker" in sel.columns
        assert "weight" in sel.columns
