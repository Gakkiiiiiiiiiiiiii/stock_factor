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


def _observed(data: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in data:
        return pd.Series(default, index=data.index, dtype=float)
    return data[column].notna().astype(float)


def _observed_flag(data: pd.DataFrame, value_column: str, flag_column: str, default: float = 0.0) -> pd.Series:
    if flag_column in data:
        return data[flag_column].astype(float).fillna(0.0)
    return _observed(data, value_column, default)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build causal V2 inputs for one symbol.

    All rolling operations are right aligned.  Missing turnover is retained as
    a calendar row and represented by a placeholder plus ``turnover_observed``;
    it is never silently replaced by a future PIT capital observation.
    """
    data = frame.sort_values("trading_date").copy()
    close = data["close"].astype(float)
    prev_close = close.shift(1)
    high, low, open_, volume, amount = [data[name].astype(float) for name in ("high", "low", "open", "volume", "amount")]
    turnover = data["turnover"].astype(float) if "turnover" in data else pd.Series(np.nan, index=data.index)
    turnover_value = turnover.fillna(0.0)
    day_range = (high - low).clip(lower=0)
    body = (close - open_).abs()
    log_volume = np.log1p(volume.clip(lower=0))
    returns = close.pct_change(fill_method=None)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()

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
    result["turnover"] = turnover_value
    result["turnover_ratio_5"] = _safe_ratio(turnover_value, turnover_value.rolling(5, min_periods=5).mean())
    result["turnover_ratio_20"] = _safe_ratio(turnover_value, turnover_value.rolling(20, min_periods=20).mean())
    result["turnover_zscore_20"] = _rolling_zscore(turnover_value, 20)
    result["true_range_close"] = _safe_ratio(tr, close)
    result["atr14_close"] = _safe_ratio(atr14, close)
    result["realized_vol_5"] = returns.rolling(5, min_periods=5).std(ddof=0)
    result["realized_vol_20"] = returns.rolling(20, min_periods=20).std(ddof=0)
    result["realized_vol_60"] = returns.rolling(60, min_periods=60).std(ddof=0)
    result["range_atr14"] = _safe_ratio(day_range, atr14)

    # Observed flags are separate from the state value.  A zero can therefore
    # mean a real false value, while a missing observation remains identifiable.
    price_observed = data[["open", "high", "low", "close"]].notna().all(axis=1).astype(float)
    volume_observed = _observed(data, "volume")
    turnover_observed = _observed_flag(data, "turnover", "turnover_observed")
    result["is_suspended"] = data["is_suspended"].astype(float) if "is_suspended" in data else ((volume <= 0) | (price_observed == 0)).astype(float)
    result["is_st"] = data["is_st"].astype(float) if "is_st" in data else 0.0
    result["is_star_st"] = data["is_star_st"].astype(float) if "is_star_st" in data else 0.0
    result["is_delisting"] = data["is_delisting"].astype(float) if "is_delisting" in data else 0.0
    result["is_limit_up"] = data["is_limit_up"].astype(float) if "is_limit_up" in data else 0.0
    result["is_limit_down"] = data["is_limit_down"].astype(float) if "is_limit_down" in data else 0.0
    result["st_observed"] = _observed_flag(data, "is_st", "st_observed")
    result["suspension_observed"] = _observed_flag(data, "is_suspended", "suspension_observed")
    result["limit_observed"] = _observed_flag(data, "is_limit_up", "limit_observed")
    result["delisting_observed"] = _observed_flag(data, "is_delisting", "delisting_observed")
    result["turnover_observed"] = turnover_observed
    result["price_observed"] = _observed_flag(data, "close", "price_observed", 1.0) * price_observed
    result["volume_observed"] = _observed_flag(data, "volume", "volume_observed", 1.0) * volume_observed
    if "listing_days" in data:
        result["listing_days_norm"] = (data["listing_days"].astype(float) / 252.0).clip(0, 20)
    else:
        result["listing_days_norm"] = np.nan

    # Quality covers the price/volume calendar row.  Turnover is a separate
    # group mask so missing PIT capital does not delete an otherwise valid day.
    valid = price_observed.astype(bool) & volume_observed.astype(bool)
    valid &= ((high >= pd.concat([open_, close], axis=1).max(axis=1)) | ~price_observed.astype(bool))
    valid &= ((low <= pd.concat([open_, close], axis=1).min(axis=1)) | ~price_observed.astype(bool))
    valid &= ((high >= low) | ~price_observed.astype(bool))
    result["quality_mask"] = valid.astype(float)
    if "quality_mask" in data:
        result["quality_mask"] = data["quality_mask"].astype(float).fillna(result["quality_mask"])

    result = result[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)
    result["quality_mask"] = result["quality_mask"].fillna(0.0)
    for column in STATE_FEATURES:
        result[column] = result[column].fillna(0.0)
    return result


def assert_feature_causality(frame: pd.DataFrame, cutoff: int) -> None:
    """Raise if changing rows after ``cutoff`` changes earlier features."""
    from pandas.testing import assert_frame_equal

    baseline = build_features(frame).iloc[:cutoff].reset_index(drop=True)
    changed = frame.copy()
    future_columns = [column for column in ("open", "high", "low", "close", "volume", "amount", "turnover") if column in changed]
    changed.loc[cutoff:, future_columns] = changed.loc[cutoff:, future_columns].astype(float) * 1000.0
    altered = build_features(changed).iloc[:cutoff].reset_index(drop=True)
    assert_frame_equal(baseline, altered, check_exact=False, rtol=1e-6, atol=1e-6)
