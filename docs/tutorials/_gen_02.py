"""Generate Tutorial 2: Mean Reversion — real data, plots."""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.10.0"}})
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

# ---------------------------------------------------------------------------
# Title & overview
# ---------------------------------------------------------------------------
md("""# Tutorial 2 — Mean Reversion

## Overview

Mean reversion is the idea that prices tend to revert to a long-run average:
stocks that have **fallen sharply** relative to their recent history are
expected to bounce back, while stocks that have **surged** are expected to
pull back.

This tutorial covers:
1. Setup and data — the same 20-stock US equity universe from Tutorial 1.
2. Short-horizon vs long-horizon mean reversion z-scores.
3. Signal comparison — z-score, RSI, Bollinger Band signals.
4. Volatility scaling — dividing signals by realised volatility.
5. Backtest comparison table across signals and rebalance frequencies.
6. Signal decay — information coefficient (IC) at multiple forward horizons.
7. Failure modes — trending markets, structural breaks, leg decomposition.
8. Summary.""")

# ---------------------------------------------------------------------------
# 1. Setup and data
# ---------------------------------------------------------------------------
md("## 1. Setup and data")

SETUP = '''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.figsize": (12, 5), "figure.dpi": 100,
                     "axes.grid": True, "grid.alpha": 0.3})

from qlab.data import YFinanceProvider, ParquetCache
from qlab.features import simple_returns, realized_volatility, rank, zscore, winsorize, demean
from qlab.features.rolling import rolling_mean, rolling_std
from qlab.alphas import mean_reversion_zscore, rsi_signal, bollinger_signal, momentum
from qlab.portfolio import equal_weight_long_short, normalize_weights
from qlab.backtest import run_backtest, BacktestConfig
from qlab.risk import performance_summary, drawdown_series

TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META",
    "JPM", "GS", "BAC",
    "JNJ", "PFE", "UNH",
    "XOM", "CVX",
    "PG", "KO", "WMT",
    "HD", "NKE",
    "CAT", "HON",
]
START, END = "2018-01-01", "2024-12-31"

provider = ParquetCache(YFinanceProvider(), cache_dir=".qlab_cache")
prices = provider.fetch(TICKERS, START, END)
close = prices["adj_close"]
print(f"Universe : {close.index.get_level_values('ticker').nunique()} stocks")
print(f"Date range: {close.index.get_level_values('date').min().date()} to "
      f"{close.index.get_level_values('date').max().date()}")
print(f"Total obs : {len(close):,}")'''

code(SETUP)

# ---------------------------------------------------------------------------
# 2. Short-horizon vs long-horizon mean reversion
# ---------------------------------------------------------------------------
md("""## 2. Short-horizon vs long-horizon mean reversion

The `mean_reversion_zscore` signal measures how far a stock's price is
from its rolling mean, expressed in standard deviations. The sign is
**negated** so that oversold stocks receive a **positive** signal.

| Lookback | Interpretation |
|----------|---------------|
| 5 days   | Very short-term microstructure reversion |
| 20 days  | Standard short-horizon MR (~1 month) |
| 60 days  | Medium-horizon MR (~3 months) |

Shorter lookbacks capture faster dislocations but also more noise.""")

code('''# Compute z-scores at three lookback horizons
zscore_5d = mean_reversion_zscore(close, lookback=5)
zscore_20d = mean_reversion_zscore(close, lookback=20)
zscore_60d = mean_reversion_zscore(close, lookback=60)

# Summary statistics
stats = pd.DataFrame({
    "5d": zscore_5d.dropna().describe(),
    "20d": zscore_20d.dropna().describe(),
    "60d": zscore_60d.dropna().describe(),
})
print("Z-score statistics by lookback:")
print(stats.round(3).to_string())''')

code('''# Cross-sectional snapshot comparison
sample_date = zscore_20d.dropna().index.get_level_values("date").unique()[120]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (label, sig) in zip(axes, [("5-day Z", zscore_5d),
                                     ("20-day Z", zscore_20d),
                                     ("60-day Z", zscore_60d)]):
    cross = sig.loc[sample_date].sort_values()
    colors = ["#d32f2f" if v < 0 else "#388e3c" for v in cross.values]
    cross.plot.barh(ax=ax, color=colors)
    ax.set_title(f"{label} -- {sample_date.date()}")
    ax.set_xlabel("Signal value")
plt.suptitle("Mean-reversion z-scores at different lookbacks", y=1.02, fontsize=14)
plt.tight_layout()
plt.show()''')

code('''# Autocorrelation: do the signals persist or mean-revert themselves?
from qlab.features.cross_section import rank as cs_rank

ac_results = {}
for label, sig in [("5d", zscore_5d), ("20d", zscore_20d), ("60d", zscore_60d)]:
    sig_clean = sig.dropna()
    r_now = cs_rank(sig_clean)
    r_lag = cs_rank(sig_clean.groupby(level="ticker").shift(5).dropna())
    common = r_now.index.intersection(r_lag.index)
    if len(common) > 0:
        corr = r_now.loc[common].groupby(level="date").corr(r_lag.loc[common])
        ac_results[label] = corr.mean()
print("Rank autocorrelation (lag-5d):")
for k, v in ac_results.items():
    print(f"  {k:>4s} lookback: {v:.3f}")''')

# ---------------------------------------------------------------------------
# 3. Signal comparison
# ---------------------------------------------------------------------------
md("""## 3. Signal comparison -- Z-score vs RSI vs Bollinger

All three signals capture mean-reversion but differ in construction:

| Signal | Mechanism | Range |
|--------|-----------|-------|
| `mean_reversion_zscore` | Negated distance from rolling mean | unbounded |
| `rsi_signal` | Rescaled RSI: (50 - RSI)/50 | [-1, 1] |
| `bollinger_signal` | Inverted Bollinger %B | roughly [-0.5, 0.5] |""")

code('''sig_zscore = mean_reversion_zscore(close, lookback=20)
sig_rsi = rsi_signal(close, period=14)
sig_boll = bollinger_signal(close, window=20, num_std=2.0)

# Cross-sectional snapshot comparison
snap_date = sig_zscore.dropna().index.get_level_values("date").unique()[200]
_snap_zscore = lambda s: (s - s.mean()) / s.std()  # simple z-score for single cross-section
snap_data = pd.DataFrame({
    "Z-score": _snap_zscore(sig_zscore.loc[snap_date]),
    "RSI": _snap_zscore(sig_rsi.loc[snap_date]),
    "Bollinger": _snap_zscore(sig_boll.loc[snap_date]),
})

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(snap_data))
w = 0.25
ax.bar(x - w, snap_data["Z-score"], width=w, label="Z-score", color="steelblue")
ax.bar(x,     snap_data["RSI"],     width=w, label="RSI",     color="crimson")
ax.bar(x + w, snap_data["Bollinger"], width=w, label="Bollinger", color="#388e3c")
ax.set_xticks(x)
ax.set_xticklabels(snap_data.index.get_level_values("ticker"), rotation=45, ha="right")
ax.set_ylabel("Cross-sectional z-score of signal")
ax.set_title(f"Signal comparison (normalised) -- {snap_date.date()}")
ax.legend()
plt.tight_layout()
plt.show()''')

code('''# Pairwise rank correlations across the whole sample
from qlab.features.cross_section import rank as cs_rank

pairs = [("Z-score", sig_zscore), ("RSI", sig_rsi), ("Bollinger", sig_boll)]
corr_matrix = pd.DataFrame(index=[p[0] for p in pairs],
                            columns=[p[0] for p in pairs], dtype=float)
for i, (n1, s1) in enumerate(pairs):
    for j, (n2, s2) in enumerate(pairs):
        s1c, s2c = s1.dropna(), s2.dropna()
        common = s1c.index.intersection(s2c.index)
        r1 = cs_rank(s1c.loc[common])
        r2 = cs_rank(s2c.loc[common])
        corr_matrix.loc[n1, n2] = r1.groupby(level="date").corr(r2).mean()

print("Average cross-sectional rank correlation between MR signals:")
print(corr_matrix.round(3).to_string())''')

# ---------------------------------------------------------------------------
# 4. Volatility scaling
# ---------------------------------------------------------------------------
md("""## 4. Volatility scaling

Mean-reversion signals can be improved by **volatility scaling**: dividing
the raw signal by realised volatility. This ensures that a 2-sigma
dislocation in a low-vol stock is not drowned out by noise in a high-vol
stock.

$$\\text{signal}^{\\text{scaled}}_i = \\frac{\\text{signal}_i}{\\sigma_i}$$""")

code('''ret = simple_returns(close)
vol = realized_volatility(ret.dropna(), window=21, annualize=False)

sig_raw = mean_reversion_zscore(close, lookback=20)
sig_raw_clean = sig_raw.dropna()
vol_aligned = vol.reindex(sig_raw_clean.index).dropna()
common_idx = sig_raw_clean.index.intersection(vol_aligned.index)
sig_scaled = sig_raw_clean.loc[common_idx] / vol_aligned.loc[common_idx].replace(0, np.nan)
sig_scaled = sig_scaled.dropna()

# Show before/after for a sample date
plot_date = sig_scaled.index.get_level_values("date").unique()[150]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

raw_snap = sig_raw_clean.loc[plot_date].sort_values()
colors_raw = ["#d32f2f" if v < 0 else "#388e3c" for v in raw_snap.values]
raw_snap.plot.barh(ax=ax1, color=colors_raw)
ax1.set_title(f"Raw MR signal -- {plot_date.date()}")
ax1.set_xlabel("Signal")

scaled_snap = sig_scaled.loc[plot_date].sort_values()
colors_sc = ["#d32f2f" if v < 0 else "#388e3c" for v in scaled_snap.values]
scaled_snap.plot.barh(ax=ax2, color=colors_sc)
ax2.set_title(f"Vol-scaled MR signal -- {plot_date.date()}")
ax2.set_xlabel("Signal / Vol")

plt.suptitle("Effect of volatility scaling on cross-sectional signal", y=1.02, fontsize=13)
plt.tight_layout()
plt.show()

print(f"Raw signal cross-sectional std:    {sig_raw_clean.groupby(level='date').std().mean():.3f}")
print(f"Scaled signal cross-sectional std: {sig_scaled.groupby(level='date').std().mean():.3f}")''')

# ---------------------------------------------------------------------------
# 5. Backtest comparison table
# ---------------------------------------------------------------------------
md("""## 5. Backtest comparison

We compare **4 signals** across **2 rebalance frequencies** (daily, weekly).

| Signal | Description |
|--------|-----------|
| MR Z-score (20d) | Standard mean-reversion z-score |
| MR Z-score vol-scaled | Z-score divided by realised vol |
| RSI | Relative Strength Index signal |
| Bollinger | Bollinger Band %B inverted |""")

code('''# Prepare all four signals
signals = {}
signals["MR_Zscore"] = mean_reversion_zscore(close, lookback=20).dropna()

# Vol-scaled
_raw = mean_reversion_zscore(close, lookback=20).dropna()
_vol = realized_volatility(simple_returns(close).dropna(), window=21, annualize=False)
_vol = _vol.reindex(_raw.index).dropna()
_common = _raw.index.intersection(_vol.index)
_scaled = (_raw.loc[_common] / _vol.loc[_common].replace(0, np.nan)).dropna()
signals["MR_VolScaled"] = _scaled

signals["RSI"] = rsi_signal(close, period=14).dropna()
signals["Bollinger"] = bollinger_signal(close, window=20, num_std=2.0).dropna()

# Build weights for each signal
all_weights = {}
for name, sig in signals.items():
    w = equal_weight_long_short(sig, long_pct=0.2, short_pct=0.2)
    w = normalize_weights(w, gross_exposure=2.0, net_exposure=0.0)
    all_weights[name] = w

# Backtest each signal x frequency
results_table = []
result_curves = {}
for freq in ["daily", "weekly"]:
    for name, w in all_weights.items():
        cfg = BacktestConfig(
            rebalance_freq=freq, commission_bps=5.0, slippage_bps=5.0,
            signal_lag=1, execution_price="open",
        )
        res = run_backtest(w, prices, config=cfg)
        perf = performance_summary(res.portfolio_returns)
        results_table.append({
            "Signal": name,
            "Frequency": freq,
            "Sharpe": perf["sharpe_ratio"],
            "Ann. Return": perf["annualized_return"],
            "Ann. Vol": perf["annualized_volatility"],
            "Max DD": perf["max_drawdown"],
            "Hit Rate": perf["hit_rate"],
        })
        result_curves[f"{name}_{freq}"] = (1 + res.portfolio_returns).cumprod()

results_df = pd.DataFrame(results_table)
print("Backtest comparison table:")
print(results_df.round(4).to_string(index=False))''')

code('''# Equity curves -- daily frequency only
fig, ax = plt.subplots(figsize=(12, 6))
colors_map = {"MR_Zscore": "steelblue", "MR_VolScaled": "navy",
              "RSI": "crimson", "Bollinger": "#388e3c"}
for name in signals.keys():
    key = f"{name}_daily"
    if key in result_curves:
        sr = results_df[(results_df["Signal"] == name) &
                        (results_df["Frequency"] == "daily")]["Sharpe"].values[0]
        result_curves[key].plot(ax=ax, label=f"{name} (SR={sr:.2f})",
                                color=colors_map.get(name, "grey"), linewidth=1.3)
ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
ax.set_ylabel("Cumulative Return")
ax.set_title("Mean-Reversion Strategies -- Equity Curves (daily rebalance)")
ax.legend(loc="best")
plt.tight_layout()
plt.show()''')

code('''# Bar chart: Sharpe by signal and frequency
fig, ax = plt.subplots(figsize=(10, 5))
pivot = results_df.pivot(index="Signal", columns="Frequency", values="Sharpe")
pivot = pivot[["daily", "weekly"]]
x = np.arange(len(pivot))
w = 0.35
ax.bar(x - w/2, pivot["daily"],  width=w, label="Daily",  color="steelblue")
ax.bar(x + w/2, pivot["weekly"], width=w, label="Weekly", color="crimson")
ax.set_xticks(x)
ax.set_xticklabels(pivot.index, rotation=20, ha="right")
ax.set_ylabel("Sharpe Ratio")
ax.set_title("Sharpe Ratio: Signal x Rebalance Frequency")
ax.legend()
ax.axhline(0, color="black", linewidth=0.5)
plt.tight_layout()
plt.show()''')

# ---------------------------------------------------------------------------
# 6. Signal decay IC analysis
# ---------------------------------------------------------------------------
md("""## 6. Signal decay -- Information Coefficient analysis

The **Information Coefficient (IC)** is the cross-sectional rank correlation
between the signal today and forward returns at some horizon. Plotting IC
against horizon reveals how quickly the signal's predictive power decays.

For mean-reversion signals we expect:
- **High IC at short horizons** (1-5 days) -- the reversion effect.
- **Decaying IC** as the horizon lengthens.
- **Possibly negative IC** at long horizons if momentum dominates.""")

code('''from qlab.features.cross_section import rank as cs_rank

horizons = [1, 5, 10, 21, 42, 63]
ic_results = {}

# Use the 20d MR z-score signal
sig = mean_reversion_zscore(close, lookback=20).dropna()
sig_ranked = cs_rank(sig)

for h in horizons:
    fwd_ret = simple_returns(close, periods=h).groupby(level="ticker").shift(-h).dropna()
    fwd_ranked = cs_rank(fwd_ret)
    common = sig_ranked.index.intersection(fwd_ranked.index)
    if len(common) > 0:
        ic = sig_ranked.loc[common].groupby(level="date").corr(fwd_ranked.loc[common])
        ic_results[h] = {"mean": ic.mean(), "std": ic.std(),
                         "ir": ic.mean() / ic.std() if ic.std() > 0 else 0}

ic_df = pd.DataFrame(ic_results).T
ic_df.index.name = "Horizon (days)"
print("IC decay analysis (MR Z-score 20d):")
print(ic_df.round(4).to_string())''')

code('''# IC decay plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart of mean IC
ax1.bar([str(h) for h in horizons],
        [ic_results[h].get("mean", 0) for h in horizons],
        color="steelblue", edgecolor="navy", linewidth=0.8)
ax1.set_xlabel("Forward return horizon (days)")
ax1.set_ylabel("Mean IC (rank correlation)")
ax1.set_title("Mean IC by horizon -- MR Z-score (20d)")
ax1.axhline(0, color="black", linewidth=0.5)

# IC for all three signals
ic_all_signals = {}
for label, sig_obj in [("Z-score", sig_zscore), ("RSI", sig_rsi), ("Bollinger", sig_boll)]:
    sig_r = cs_rank(sig_obj.dropna())
    ic_by_h = []
    for h in horizons:
        fwd = simple_returns(close, periods=h).groupby(level="ticker").shift(-h).dropna()
        fwd_r = cs_rank(fwd)
        common = sig_r.index.intersection(fwd_r.index)
        if len(common) > 0:
            ic_val = sig_r.loc[common].groupby(level="date").corr(fwd_r.loc[common]).mean()
        else:
            ic_val = 0.0
        ic_by_h.append(ic_val)
    ic_all_signals[label] = ic_by_h

colors_ic = {"Z-score": "steelblue", "RSI": "crimson", "Bollinger": "#388e3c"}
for label, vals in ic_all_signals.items():
    ax2.plot(horizons, vals, marker="o", label=label, color=colors_ic[label], linewidth=1.5)
ax2.set_xlabel("Forward return horizon (days)")
ax2.set_ylabel("Mean IC")
ax2.set_title("IC decay comparison across MR signals")
ax2.axhline(0, color="black", linewidth=0.5)
ax2.legend()

plt.tight_layout()
plt.show()''')

# ---------------------------------------------------------------------------
# 7. Failure modes
# ---------------------------------------------------------------------------
md("""## 7. Failure modes

### 7a. Trending markets -- Momentum vs Mean Reversion

Mean reversion profits when prices snap back; momentum profits when trends
persist. These are **natural adversaries**. In strongly trending markets,
MR strategies suffer as they sell winners and buy losers too early.""")

code('''# Backtest momentum alongside MR for comparison
mom_sig = momentum(close, lookback=252, skip=21).dropna()
mom_w = equal_weight_long_short(mom_sig, long_pct=0.2, short_pct=0.2)
mom_w = normalize_weights(mom_w, gross_exposure=2.0, net_exposure=0.0)

mr_w = all_weights["MR_Zscore"]

cfg_compare = BacktestConfig(
    rebalance_freq="weekly", commission_bps=5.0, slippage_bps=5.0,
    signal_lag=1, execution_price="open",
)

res_mom = run_backtest(mom_w, prices, config=cfg_compare)
res_mr = run_backtest(mr_w, prices, config=cfg_compare)

cum_mom = (1 + res_mom.portfolio_returns).cumprod()
cum_mr = (1 + res_mr.portfolio_returns).cumprod()

# Rolling 63-day return correlation
rolling_corr = res_mom.portfolio_returns.rolling(63, min_periods=63).corr(
    res_mr.portfolio_returns
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

cum_mom.plot(ax=ax1, color="navy", label="Momentum", linewidth=1.3)
cum_mr.plot(ax=ax1, color="crimson", label="Mean Reversion", linewidth=1.3)
ax1.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
ax1.set_ylabel("Cumulative Return")
ax1.set_title("Momentum vs Mean Reversion -- Equity Curves")
ax1.legend()

rolling_corr.plot(ax=ax2, color="steelblue", linewidth=1)
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_ylabel("Rolling 63d correlation")
ax2.set_title("Return correlation between Momentum and MR")
ax2.fill_between(rolling_corr.index, rolling_corr.values, 0,
                 where=rolling_corr.values < 0, alpha=0.2, color="crimson")
ax2.fill_between(rolling_corr.index, rolling_corr.values, 0,
                 where=rolling_corr.values >= 0, alpha=0.2, color="steelblue")

plt.tight_layout()
plt.show()

perf_mom = performance_summary(res_mom.portfolio_returns)
perf_mr = performance_summary(res_mr.portfolio_returns)
print(f"Momentum Sharpe:       {perf_mom['sharpe_ratio']:.3f}")
print(f"Mean Reversion Sharpe: {perf_mr['sharpe_ratio']:.3f}")
print(f"Daily return correlation: {res_mom.portfolio_returns.corr(res_mr.portfolio_returns):.3f}")''')

code('''# Scatter plot: MR daily returns vs Momentum daily returns
fig, ax = plt.subplots(figsize=(7, 7))
mr_ret = res_mr.portfolio_returns.dropna()
mom_ret = res_mom.portfolio_returns.reindex(mr_ret.index).dropna()
mr_aligned = mr_ret.reindex(mom_ret.index)

ax.scatter(mom_ret.values * 100, mr_aligned.values * 100, alpha=0.25, s=8, color="navy")
ax.axhline(0, color="grey", linewidth=0.5)
ax.axvline(0, color="grey", linewidth=0.5)
ax.set_xlabel("Momentum Daily Return (%)")
ax.set_ylabel("Mean Reversion Daily Return (%)")
ax.set_title("MR vs Momentum -- daily return scatter")

# Quadrant counts
q1 = ((mom_ret > 0) & (mr_aligned > 0)).sum()
q2 = ((mom_ret < 0) & (mr_aligned > 0)).sum()
q3 = ((mom_ret < 0) & (mr_aligned < 0)).sum()
q4 = ((mom_ret > 0) & (mr_aligned < 0)).sum()
n = len(mom_ret)
ax.text(0.95, 0.95, f"Both +: {q1/n:.0%}", transform=ax.transAxes, ha="right", va="top", fontsize=9)
ax.text(0.05, 0.95, f"Mom-, MR+: {q2/n:.0%}", transform=ax.transAxes, ha="left", va="top", fontsize=9)
ax.text(0.05, 0.05, f"Both -: {q3/n:.0%}", transform=ax.transAxes, ha="left", va="bottom", fontsize=9)
ax.text(0.95, 0.05, f"Mom+, MR-: {q4/n:.0%}", transform=ax.transAxes, ha="right", va="bottom", fontsize=9)

plt.tight_layout()
plt.show()''')

md("""### 7b. Structural breaks, fat tails, and asymmetry

**Structural breaks**: Mean reversion assumes prices fluctuate around a
stable mean. When a stock undergoes a fundamental regime change (e.g. a
permanent earnings shock), the old mean is no longer relevant and the MR
signal generates persistent losses buying a falling knife.

**Fat tails**: MR signals based on z-scores assume normally distributed
deviations. In practice, stock returns have fat tails, meaning extreme
dislocations occur more often than expected. An "oversold" stock at -3 sigma
can easily move to -5 sigma before reverting (if it reverts at all).

**Asymmetry between legs**: The short leg (selling recent winners) tends to
be more dangerous than the long leg (buying recent losers). Winners can
keep winning (momentum), while losers are at least bounded by zero. This
creates a structural drag on the short side of MR portfolios.""")

# ---------------------------------------------------------------------------
# 7c. Leg decomposition
# ---------------------------------------------------------------------------
md("### 7c. Long-leg vs short-leg decomposition")

code('''# Decompose MR portfolio into long and short legs
mr_positions = res_mr.positions  # wide-form: date x ticker
asset_returns = prices["close"].unstack("ticker").pct_change().fillna(0.0)
asset_returns = asset_returns.reindex(index=mr_positions.index,
                                       columns=mr_positions.columns).fillna(0.0)

# Long leg: positions > 0
long_pos = mr_positions.clip(lower=0)
short_pos = mr_positions.clip(upper=0)

long_ret = (long_pos * asset_returns).sum(axis=1)
short_ret = (short_pos * asset_returns).sum(axis=1)
total_ret = long_ret + short_ret

cum_long = (1 + long_ret).cumprod()
cum_short = (1 + short_ret).cumprod()
cum_total = (1 + total_ret).cumprod()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

cum_long.plot(ax=ax1, color="#388e3c", label="Long leg", linewidth=1.3)
cum_short.plot(ax=ax1, color="#d32f2f", label="Short leg", linewidth=1.3)
cum_total.plot(ax=ax1, color="navy", label="Total portfolio", linewidth=1.5, linestyle="--")
ax1.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
ax1.set_ylabel("Cumulative Return")
ax1.set_title("MR Portfolio -- Long vs Short Leg Decomposition")
ax1.legend()

# Rolling 63d annualised return for each leg
roll_long = long_ret.rolling(63).mean() * 252
roll_short = short_ret.rolling(63).mean() * 252
roll_long.plot(ax=ax2, color="#388e3c", label="Long leg (ann.)", linewidth=1)
roll_short.plot(ax=ax2, color="#d32f2f", label="Short leg (ann.)", linewidth=1)
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_ylabel("Rolling 63d Ann. Return")
ax2.set_title("Rolling P&L by Leg")
ax2.legend()

plt.tight_layout()
plt.show()

perf_long = performance_summary(long_ret)
perf_short = performance_summary(short_ret)
print(f"Long leg  -- Ann. return: {perf_long['annualized_return']:+.4f}, "
      f"Sharpe: {perf_long['sharpe_ratio']:.3f}")
print(f"Short leg -- Ann. return: {perf_short['annualized_return']:+.4f}, "
      f"Sharpe: {perf_short['sharpe_ratio']:.3f}")''')

# ---------------------------------------------------------------------------
# 7d. Drawdown analysis
# ---------------------------------------------------------------------------
md("### 7d. MR strategy drawdown")

code('''dd = drawdown_series(res_mr.portfolio_returns)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
cum_mr.plot(ax=ax1, color="navy", linewidth=1.5)
ax1.set_ylabel("Cumulative Return")
ax1.set_title("Mean-Reversion Strategy -- Equity Curve")
ax1.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)

dd.plot(ax=ax2, color="crimson", linewidth=1)
ax2.fill_between(dd.index, dd.values, 0, alpha=0.3, color="crimson")
ax2.set_ylabel("Drawdown")
ax2.set_title("Underwater Chart")
plt.tight_layout()
plt.show()

print(f"Max drawdown: {dd.min():.4f}")
print(f"Time in drawdown (>1%): {(dd < -0.01).mean():.1%}")''')

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
md("""## 8. Summary

| Concept | Key takeaway |
|---------|-------------|
| Signal | Price distance from rolling mean (negated z-score) |
| Lookback | 20d is standard; shorter = more signal, more noise |
| Alternatives | RSI and Bollinger capture similar effect with different mechanics |
| Volatility scaling | Dividing by realised vol improves cross-sectional comparability |
| Rebalancing | Higher frequency generally helps MR (signals are short-lived) |
| IC decay | Strongest at short horizons (1-5 days), fades rapidly |
| Failure modes | Trending markets, structural breaks, short-leg drag |
| Complementarity | MR and momentum are natural hedges when combined |""")

# ---------------------------------------------------------------------------
# Write the notebook
# ---------------------------------------------------------------------------
nb.cells = cells
nbf.write(nb, "/scratch/scratch-lxu/qlab/docs/tutorials/02_mean_reversion.ipynb")
print("OK")
