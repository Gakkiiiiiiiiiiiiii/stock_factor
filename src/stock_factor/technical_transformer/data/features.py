from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import CONTINUOUS_FEATURES, FEATURE_NAMES, STATE_FEATURES


EPS = 1e-8


def _safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.astype(float) / b.replace(0, np.nan).astype(float)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0).replace(0, np.nan)
    return (series - mean) / (std + EPS)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build causal raw/normalized technical inputs for one symbol.

    Every rolling operation is right-aligned.  No feature reads a row after
    the current as-of date.
    """
    data = frame.sort_values("trading_date").copy()
    close = data["close"].astype(float)
    prev_close = close.shift(1)
    high, low, open_, volume, amount, turnover = [data[name].astype(float) for name in ("high", "low", "open", "volume", "amount", "turnover")]
    day_range = (high - low).clip(lower=0)
    body = (close - open_).abs()
    log_volume = np.log1p(volume.clip(lower=0))
    returns = close.pct_change(fill_method=None)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()
    sma20, sma60 = close.rolling(20, min_periods=20).mean(), close.rolling(60, min_periods=60).mean()
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()

    result = pd.DataFrame(index=data.index)
    result["ret_1"] = returns
    result["ret_3"] = close.pct_change(3, fill_method=None)
    result["ret_5"] = close.pct_change(5, fill_method=None)
    result["ret_10"] = close.pct_change(10, fill_method=None)
    result["open_prev_close"] = _safe_ratio(open_, prev_close) - 1
    result["high_prev_close"] = _safe_ratio(high, prev_close) - 1
    result["low_prev_close"] = _safe_ratio(low, prev_close) - 1
    result["close_prev_close"] = _safe_ratio(close, prev_close) - 1
    result["intraday_range_prev_close"] = _safe_ratio(day_range, prev_close)
    result["intraday_body_prev_close"] = _safe_ratio(close - open_, prev_close)
    result["body_ratio"] = _safe_ratio(body, day_range + EPS)
    result["upper_shadow_ratio"] = _safe_ratio(high - pd.concat([open_, close], axis=1).max(axis=1), day_range + EPS)
    result["lower_shadow_ratio"] = _safe_ratio(pd.concat([open_, close], axis=1).min(axis=1) - low, day_range + EPS)
    result["range_position"] = _safe_ratio(close - low, day_range + EPS)
    result["gap_ratio"] = _safe_ratio(open_ - prev_close, prev_close)
    result["log1p_volume"] = log_volume
    for window in (5, 10, 20, 60):
        result[f"volume_ratio_{window}"] = _safe_ratio(volume, volume.rolling(window, min_periods=window).mean())
    result["volume_zscore_20"] = _rolling_zscore(log_volume, 20)
    result["volume_zscore_60"] = _rolling_zscore(log_volume, 60)
    result["volume_change_1"] = volume.pct_change(fill_method=None)
    result["volume_change_5"] = volume.pct_change(5, fill_method=None)
    result["amount_ratio_5"] = _safe_ratio(amount, amount.rolling(5, min_periods=5).mean())
    result["amount_ratio_20"] = _safe_ratio(amount, amount.rolling(20, min_periods=20).mean())
    result["turnover"] = turnover
    result["turnover_ratio_5"] = _safe_ratio(turnover, turnover.rolling(5, min_periods=5).mean())
    result["turnover_ratio_20"] = _safe_ratio(turnover, turnover.rolling(20, min_periods=20).mean())
    result["turnover_zscore_20"] = _rolling_zscore(turnover, 20)
    result["true_range_close"] = _safe_ratio(tr, close)
    result["atr14_close"] = _safe_ratio(atr14, close)
    result["realized_vol_5"] = returns.rolling(5, min_periods=5).std(ddof=0)
    result["realized_vol_20"] = returns.rolling(20, min_periods=20).std(ddof=0)
    result["realized_vol_60"] = returns.rolling(60, min_periods=60).std(ddof=0)
    result["range_atr14"] = _safe_ratio(day_range, atr14)
    result["close_sma20"] = _safe_ratio(close, sma20) - 1
    result["close_sma60"] = _safe_ratio(close, sma60) - 1
    result["close_ema20"] = _safe_ratio(close, ema20) - 1

    for column in STATE_FEATURES:
        if column in data:
            result[column] = data[column].astype(float)
        else:
            result[column] = 0.0
    if "listing_days" in data:
        result["listing_days_norm"] = (data["listing_days"].astype(float) / 252.0).clip(0, 20)
    else:
        result["listing_days_norm"] = np.nan
    if "quality_mask" not in data:
        valid = data[["open", "high", "low", "close", "volume", "amount", "turnover"]].notna().all(axis=1)
        valid &= high >= pd.concat([open_, close], axis=1).max(axis=1)
        valid &= low <= pd.concat([open_, close], axis=1).min(axis=1)
        valid &= high >= low
        result["quality_mask"] = valid.astype(float)

    result = result[FEATURE_NAMES]
    result = result.replace([np.inf, -np.inf], np.nan)
    # Warm-up NaNs are expected and are made explicit to the dataset mask.
    result["quality_mask"] = result["quality_mask"].fillna(0.0)
    result["state_observed"] = result["state_observed"].fillna(0.0)
    return result
