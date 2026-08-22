from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import ALL_LABELS, BOLL_LABELS, EVENT_LABELS, LABEL_SCHEMA, MA_LABELS, PHASE_LABELS, WYCKOFF_PRIMITIVE_LABELS


EPS = 1e-8


def _clip01(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)


def _sigmoid(series: pd.Series, scale: float = 1.0) -> pd.Series:
    values = np.clip(series.astype(float).fillna(0.0).to_numpy() / scale, -30, 30)
    return pd.Series(1.0 / (1.0 + np.exp(-values)), index=series.index)


def _days_since(event: pd.Series) -> pd.Series:
    days: list[float] = []
    since = 999.0
    for value in event.fillna(0).astype(bool):
        since = 0.0 if value else since + 1.0
        days.append(min(since, 120.0) / 120.0)
    return pd.Series(days, index=event.index)


def _softmax(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.to_numpy(dtype=float)
    values -= np.nanmax(values, axis=1, keepdims=True)
    exp = np.exp(np.clip(values, -30, 30))
    probs = exp / (exp.sum(axis=1, keepdims=True) + EPS)
    return pd.DataFrame(probs, index=frame.index, columns=frame.columns)


def build_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Causal MA/Bollinger/Wyckoff-style weak labels for one symbol."""
    data = frame.sort_values("trading_date").copy()
    close = data["close"].astype(float)
    high, low, open_ = [data[name].astype(float) for name in ("high", "low", "open")]
    volume, amount, turnover = [data[name].astype(float) for name in ("volume", "amount", "turnover")]
    returns = close.pct_change().fillna(0.0)
    tr = pd.concat([(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=1).mean().replace(0, np.nan)
    ma = {window: close.rolling(window, min_periods=window).mean() for window in (5, 10, 20, 30, 60, 120)}
    result = pd.DataFrame(index=data.index)
    for window in (5, 10, 20, 30, 60, 120):
        result[f"ma{window}_slope"] = ma[window].pct_change(5)
    for window in (5, 20, 60, 120):
        result[f"close_ma{window}_distance"] = close / (ma[window] + EPS) - 1
    result["bull_alignment_score"] = ((ma[5] > ma[10]).astype(float) + (ma[10] > ma[20]).astype(float) + (ma[20] > ma[60]).astype(float)) / 3.0
    result["bear_alignment_score"] = ((ma[5] < ma[10]).astype(float) + (ma[10] < ma[20]).astype(float) + (ma[20] < ma[60]).astype(float)) / 3.0
    result["ma_trend_strength"] = _clip01(close.pct_change(20).abs() / (returns.rolling(20, min_periods=1).std(ddof=0) * np.sqrt(20) + EPS))
    normalized = pd.concat([ma[5] / close, ma[10] / close, ma[20] / close, ma[60] / close], axis=1)
    compression = normalized.std(axis=1, ddof=0)
    result["compression_score"] = _clip01(1.0 - compression / 0.08)
    result["ma_expansion_score"] = _clip01(compression / 0.08)
    cross_5_20 = ((ma[5] > ma[20]) & (ma[5].shift(1) <= ma[20].shift(1))).astype(float)
    cross_10_20 = ((ma[10] > ma[20]) & (ma[10].shift(1) <= ma[20].shift(1))).astype(float)
    cross_20_60 = ((ma[20] > ma[60]) & (ma[20].shift(1) <= ma[60].shift(1))).astype(float)
    result["cross_5_20"], result["cross_10_20"], result["cross_20_60"] = cross_5_20, cross_10_20, cross_20_60
    result["days_since_cross_5_20"] = _days_since((cross_5_20 > 0) | (cross_5_20.shift(1) < 0))
    result["days_since_cross_20_60"] = _days_since((cross_20_60 > 0) | (cross_20_60.shift(1) < 0))

    mid = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=0)
    upper, lower = mid + 2 * std, mid - 2 * std
    bandwidth = (upper - lower) / (mid.abs() + EPS)
    result["percent_b"] = (close - lower) / (upper - lower + EPS)
    result["boll_zscore"] = (close - mid) / (std + EPS)
    result["bandwidth"] = bandwidth
    result["bandwidth_delta_5"] = bandwidth.diff(5)
    result["bandwidth_delta_20"] = bandwidth.diff(20)
    result["bandwidth_percentile"] = bandwidth.rolling(120, min_periods=20).rank(pct=True)
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
    result["trend_direction"] = np.tanh(trend20.fillna(0) * 8)
    result["wyckoff_trend_strength"] = _clip01(trend20.abs() / (returns.rolling(20, min_periods=5).std(ddof=0) * np.sqrt(20) + EPS))
    result["trading_range_score"] = _clip01(1.0 - trend20.abs() / (path + EPS))
    result["range_position"] = _clip01(range_position)
    result["support_distance"] = _clip01((close - rolling_low) / (atr14 + EPS) / 5.0)
    result["resistance_distance"] = _clip01((rolling_high - close) / (atr14 + EPS) / 5.0)
    result["breakout_strength"] = _clip01((close - rolling_high) / (atr14 + EPS))
    result["breakdown_strength"] = _clip01((rolling_low - close) / (atr14 + EPS))
    result["false_breakout_score"] = _clip01(((high > rolling_high) & (close < rolling_high)).astype(float) + ((low < rolling_low) & (close > rolling_low)).astype(float))
    v20 = volume / (volume.rolling(20, min_periods=5).mean() + EPS)
    result["volume_expansion"] = _clip01((v20 - 1.0) / 2.0)
    result["volume_contraction"] = _clip01((1.0 - v20) / 0.8)
    effort = (v20 + turnover / (turnover.rolling(20, min_periods=5).mean() + EPS)) / 2.0
    result["effort_result_score"] = _clip01(effort / (returns.abs() + tr / (close + EPS) + 0.02))
    result["effort_result_divergence"] = np.tanh((effort - 1.0) - (returns.abs() + tr / (close + EPS)) * 10)
    body = close - open_
    position = (close - low) / (high - low + EPS)
    result["demand_pressure_proxy"] = _clip01(0.5 + 0.25 * np.tanh(returns * 10) + 0.2 * (position - 0.5) + 0.1 * _clip01((v20 - 1) / 2))
    result["supply_pressure_proxy"] = _clip01(0.5 - 0.25 * np.tanh(returns * 10) + 0.2 * (0.5 - position) + 0.1 * _clip01((v20 - 1) / 2))

    phase_scores = pd.DataFrame({
        "accumulation_like": 1.2 * result["trading_range_score"] + 0.8 * result["demand_pressure_proxy"] + 0.3 * (1 - result["wyckoff_trend_strength"]),
        "markup": 1.2 * _clip01(result["trend_direction"]) + 0.8 * result["bull_alignment_score"] + 0.3 * result["demand_pressure_proxy"],
        "distribution_like": 1.2 * result["trading_range_score"] + 0.8 * result["supply_pressure_proxy"] + 0.3 * (1 - result["wyckoff_trend_strength"]),
        "markdown": 1.2 * _clip01(-result["trend_direction"]) + 0.8 * result["bear_alignment_score"] + 0.3 * result["supply_pressure_proxy"],
        "transition": 1.0 - result["trading_range_score"] * 0.5 - result["wyckoff_trend_strength"] * 0.5,
    }, index=data.index)
    phase = _softmax(phase_scores)
    result[PHASE_LABELS] = phase

    spring = ((low < rolling_low) & (close > rolling_low) & (result["trading_range_score"] > 0.25)).astype(float)
    upthrust = ((high > rolling_high) & (close < rolling_high) & (result["trading_range_score"] > 0.25)).astype(float)
    result["spring_score"] = _clip01(spring * (0.5 + 0.5 * result["demand_pressure_proxy"]))
    result["upthrust_score"] = _clip01(upthrust * (0.5 + 0.5 * result["supply_pressure_proxy"]))
    result["sc_score"] = _clip01((result["breakdown_strength"] * 0.4 + result["volume_expansion"] * 0.3 + result["supply_pressure_proxy"] * 0.3) * (1 - result["wyckoff_trend_strength"] * 0.25))
    result["bc_score"] = _clip01((result["breakout_strength"] * 0.4 + result["volume_expansion"] * 0.3 + result["demand_pressure_proxy"] * 0.3) * (1 - result["wyckoff_trend_strength"] * 0.25))
    result["sos_score"] = _clip01(result["breakout_strength"] * 0.45 + result["demand_pressure_proxy"] * 0.35 + result["volume_expansion"] * 0.2)
    result["sow_score"] = _clip01(result["breakdown_strength"] * 0.45 + result["supply_pressure_proxy"] * 0.35 + result["volume_expansion"] * 0.2)
    result = result[ALL_LABELS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return result
