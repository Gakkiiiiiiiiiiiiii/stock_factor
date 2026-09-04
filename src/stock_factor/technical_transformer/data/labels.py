from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schemas import ALL_LABELS, PHASE_LABELS

EPS = 1e-8


@dataclass(frozen=True)
class TechnicalLabels:
    """Label values plus a per-target validity mask.

    ``__getattr__``/``__getitem__`` keep the old DataFrame-shaped API usable
    while making validity explicit for the dataset and loss functions.
    """

    values: pd.DataFrame
    valid: pd.DataFrame

    def __getattr__(self, name: str):
        return getattr(self.values, name)

    def __getitem__(self, key):
        return self.values[key]


def _clip01(series: pd.Series) -> pd.Series:
    # Keep NaN until the final value/mask split.  Filling here would make an
    # unavailable target indistinguishable from a real zero target.
    return series.replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0)


def _sigmoid(series: pd.Series, scale: float = 1.0) -> pd.Series:
    values = np.clip(series.astype(float).fillna(0.0).to_numpy() / scale, -30, 30)
    return pd.Series(1.0 / (1.0 + np.exp(-values)), index=series.index)


def _days_since(event: pd.Series) -> pd.Series:
    """Return normalized elapsed trading days since the last signed event."""
    days: list[float] = []
    since = 999.0
    for value in event.fillna(0).astype(bool):
        since = 0.0 if value else since + 1.0
        days.append(min(since, 120.0) / 120.0)
    return pd.Series(days, index=event.index)


def _softmax(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.to_numpy(dtype=float)
    available = np.isfinite(values).all(axis=1)
    safe = np.where(np.isfinite(values), values, 0.0)
    safe -= safe.max(axis=1, keepdims=True)
    exp = np.exp(np.clip(safe, -30, 30))
    probs = exp / (exp.sum(axis=1, keepdims=True) + EPS)
    probs[~available] = np.nan
    return pd.DataFrame(probs, index=frame.index, columns=frame.columns)


def build_labels(frame: pd.DataFrame) -> TechnicalLabels:
    """Causal MA/Bollinger/Wyckoff-style weak labels for one symbol."""
    data = frame.sort_values("trading_date").copy()
    close = data["close"].astype(float)
    high, low, open_ = [data[name].astype(float) for name in ("high", "low", "open")]
    volume, amount, turnover = [data[name].astype(float) for name in ("volume", "amount", "turnover")]
    turnover_observed = data["turnover_observed"].astype(bool) if "turnover_observed" in data else turnover.notna()
    turnover_for_label = turnover.where(turnover_observed, np.nan)
    returns = close.pct_change(fill_method=None).fillna(0.0)
    tr = pd.concat([(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(
        axis=1
    )
    atr14 = tr.rolling(14, min_periods=1).mean().replace(0, np.nan)
    ma = {window: close.rolling(window, min_periods=window).mean() for window in (5, 10, 20, 30, 60, 120)}
    result = pd.DataFrame(index=data.index)
    for window in (5, 10, 20, 30, 60, 120):
        result[f"ma{window}_slope"] = ma[window].pct_change(5, fill_method=None)
    for window in (5, 20, 60, 120):
        result[f"close_ma{window}_distance"] = close / (ma[window] + EPS) - 1
    result["bull_alignment_score"] = (
        (ma[5] > ma[10]).astype(float) + (ma[10] > ma[20]).astype(float) + (ma[20] > ma[60]).astype(float)
    ) / 3.0
    result["bear_alignment_score"] = (
        (ma[5] < ma[10]).astype(float) + (ma[10] < ma[20]).astype(float) + (ma[20] < ma[60]).astype(float)
    ) / 3.0
    result["ma_trend_strength"] = _clip01(
        close.pct_change(20).abs() / (returns.rolling(20, min_periods=1).std(ddof=0) * np.sqrt(20) + EPS)
    )
    normalized = pd.concat([ma[5] / close, ma[10] / close, ma[20] / close, ma[60] / close], axis=1)
    compression = normalized.std(axis=1, ddof=0)
    result["compression_score"] = _clip01(1.0 - compression / 0.08)
    result["ma_expansion_score"] = _clip01(compression / 0.08)

    def signed_cross(short: int, long: int) -> tuple[pd.Series, pd.Series]:
        previous = ma[short].shift(1) - ma[long].shift(1)
        current = ma[short] - ma[long]
        available = current.notna() & previous.notna()
        bull = ((current > 0) & (previous <= 0)).astype(float).where(available)
        bear = ((current < 0) & (previous >= 0)).astype(float).where(available)
        return bull, bear

    bull_5_20, bear_5_20 = signed_cross(5, 20)
    bull_10_20, bear_10_20 = signed_cross(10, 20)
    bull_20_60, bear_20_60 = signed_cross(20, 60)
    result["bull_cross_5_20"], result["bear_cross_5_20"] = bull_5_20, bear_5_20
    result["bull_cross_10_20"], result["bear_cross_10_20"] = bull_10_20, bear_10_20
    result["bull_cross_20_60"], result["bear_cross_20_60"] = bull_20_60, bear_20_60
    result["days_since_bull_cross_5_20"] = _days_since(bull_5_20.fillna(0) > 0).where(bull_5_20.notna())
    result["days_since_bear_cross_5_20"] = _days_since(bear_5_20.fillna(0) > 0).where(bear_5_20.notna())
    result["days_since_bull_cross_20_60"] = _days_since(bull_20_60.fillna(0) > 0).where(bull_20_60.notna())
    result["days_since_bear_cross_20_60"] = _days_since(bear_20_60.fillna(0) > 0).where(bear_20_60.notna())

    mid = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=0)
    upper, lower = mid + 2 * std, mid - 2 * std
    bandwidth = (upper - lower) / (mid.abs() + EPS)
    result["percent_b"] = (close - lower) / (upper - lower + EPS)
    result["boll_zscore"] = (close - mid) / (std + EPS)
    result["bandwidth"] = bandwidth
    result["bandwidth_delta_5"] = bandwidth.diff(5)
    result["bandwidth_delta_20"] = bandwidth.diff(20)
    result["bandwidth_percentile"] = bandwidth.rolling(120, min_periods=120).rank(pct=True)
    result["squeeze_score"] = 1.0 - result["bandwidth_percentile"]
    result["boll_expansion_score"] = _clip01(result["bandwidth_delta_5"] / 0.05)
    result["upper_break_strength"] = _clip01((close - upper) / (atr14 + EPS))
    result["lower_break_strength"] = _clip01((lower - close) / (atr14 + EPS))

    rolling_high = high.rolling(40, min_periods=20).max().shift(1)
    rolling_low = low.rolling(40, min_periods=20).min().shift(1)
    range_width = rolling_high - rolling_low
    range_position = (close - rolling_low) / (range_width + EPS)
    trend20 = close.pct_change(20)
    path = returns.abs().rolling(20, min_periods=5).sum()
    result["trend_direction"] = np.tanh(trend20 * 8)
    result["wyckoff_trend_strength"] = _clip01(
        trend20.abs() / (returns.rolling(20, min_periods=5).std(ddof=0) * np.sqrt(20) + EPS)
    )
    result["trading_range_score"] = _clip01(1.0 - trend20.abs() / (path + EPS))
    result["range_position"] = _clip01(range_position)
    result["support_distance"] = _clip01((close - rolling_low) / (atr14 + EPS) / 5.0)
    result["resistance_distance"] = _clip01((rolling_high - close) / (atr14 + EPS) / 5.0)
    result["breakout_strength"] = _clip01((close - rolling_high) / (atr14 + EPS))
    result["breakdown_strength"] = _clip01((rolling_low - close) / (atr14 + EPS))
    result["false_breakout_score"] = _clip01(
        ((high > rolling_high) & (close < rolling_high)).astype(float)
        + ((low < rolling_low) & (close > rolling_low)).astype(float)
    )
    v20 = volume / (volume.rolling(20, min_periods=5).mean() + EPS)
    result["volume_expansion"] = _clip01((v20 - 1.0) / 2.0)
    result["volume_contraction"] = _clip01((1.0 - v20) / 0.8)
    effort = (v20 + turnover_for_label / (turnover_for_label.rolling(20, min_periods=5).mean() + EPS)) / 2.0
    result["effort_result_score"] = _clip01(effort / (returns.abs() + tr / (close + EPS) + 0.02))
    result["effort_result_divergence"] = np.tanh((effort - 1.0) - (returns.abs() + tr / (close + EPS)) * 10)
    position = (close - low) / (high - low + EPS)
    result["demand_pressure_proxy"] = _clip01(
        0.5 + 0.25 * np.tanh(returns * 10) + 0.2 * (position - 0.5) + 0.1 * _clip01((v20 - 1) / 2)
    )
    result["supply_pressure_proxy"] = _clip01(
        0.5 - 0.25 * np.tanh(returns * 10) + 0.2 * (0.5 - position) + 0.1 * _clip01((v20 - 1) / 2)
    )

    phase_scores = pd.DataFrame(
        {
            "accumulation_like": 1.2 * result["trading_range_score"]
            + 0.8 * result["demand_pressure_proxy"]
            + 0.3 * (1 - result["wyckoff_trend_strength"]),
            "markup": 1.2 * _clip01(result["trend_direction"])
            + 0.8 * result["bull_alignment_score"]
            + 0.3 * result["demand_pressure_proxy"],
            "distribution_like": 1.2 * result["trading_range_score"]
            + 0.8 * result["supply_pressure_proxy"]
            + 0.3 * (1 - result["wyckoff_trend_strength"]),
            "markdown": 1.2 * _clip01(-result["trend_direction"])
            + 0.8 * result["bear_alignment_score"]
            + 0.3 * result["supply_pressure_proxy"],
            "transition": 1.0 - result["trading_range_score"] * 0.5 - result["wyckoff_trend_strength"] * 0.5,
        },
        index=data.index,
    )
    phase = _softmax(phase_scores)
    result[PHASE_LABELS] = phase

    spring = ((low < rolling_low) & (close > rolling_low) & (result["trading_range_score"] > 0.25)).astype(float)
    upthrust = ((high > rolling_high) & (close < rolling_high) & (result["trading_range_score"] > 0.25)).astype(float)
    result["spring_score"] = _clip01(spring * (0.5 + 0.5 * result["demand_pressure_proxy"]))
    result["upthrust_score"] = _clip01(upthrust * (0.5 + 0.5 * result["supply_pressure_proxy"]))
    result["sc_score"] = _clip01(
        (result["breakdown_strength"] * 0.4 + result["volume_expansion"] * 0.3 + result["supply_pressure_proxy"] * 0.3)
        * (1 - result["wyckoff_trend_strength"] * 0.25)
    )
    result["bc_score"] = _clip01(
        (result["breakout_strength"] * 0.4 + result["volume_expansion"] * 0.3 + result["demand_pressure_proxy"] * 0.3)
        * (1 - result["wyckoff_trend_strength"] * 0.25)
    )
    result["sos_score"] = _clip01(
        result["breakout_strength"] * 0.45 + result["demand_pressure_proxy"] * 0.35 + result["volume_expansion"] * 0.2
    )
    result["sow_score"] = _clip01(
        result["breakdown_strength"] * 0.45 + result["supply_pressure_proxy"] * 0.35 + result["volume_expansion"] * 0.2
    )
    result = result[ALL_LABELS].replace([np.inf, -np.inf], np.nan)
    valid = result.notna()
    # A placeholder turnover must not become a synthetic negative/positive
    # effort observation.  Price/volume-only targets remain usable.
    for name in ("effort_result_score", "effort_result_divergence"):
        valid[name] &= turnover_observed
    values = result.fillna(0.0)
    return TechnicalLabels(values=values, valid=valid.astype(bool))


def assert_label_causality(frame: pd.DataFrame, cutoff: int) -> None:
    """Raise if changing rows after ``cutoff`` changes earlier labels."""
    from pandas.testing import assert_frame_equal

    baseline = build_labels(frame).values.iloc[:cutoff].reset_index(drop=True)
    changed = frame.copy()
    future_columns = [
        column for column in ("open", "high", "low", "close", "volume", "amount", "turnover") if column in changed
    ]
    changed.loc[cutoff:, future_columns] = changed.loc[cutoff:, future_columns].astype(float) * 1000.0
    altered = build_labels(changed).values.iloc[:cutoff].reset_index(drop=True)
    assert_frame_equal(baseline, altered, check_exact=False, rtol=1e-6, atol=1e-6)
