from __future__ import annotations

import re
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from stock_factor.engine.vocab import BINARY_OPS, CS_OPS, TERNARY_OPS, TS_BINARY_OPS, TS_WINDOWS, UNARY_OPS

_EPS = 1e-12
_TS_RE = re.compile(
    r"^(ts_mean|ts_std|ts_max|ts_min|ts_delta|ts_delay|ts_rank|ts_sum|ts_corr|ts_cov|"
    r"ts_argmax|ts_argmin|decay_linear|count)_(\d+)$"
)


def parse_ts_token(token: str) -> tuple[str, int] | None:
    match = _TS_RE.match(token)
    if not match or int(match.group(2)) not in TS_WINDOWS:
        return None
    return match.group(1), int(match.group(2))


def _rolling(x: np.ndarray, window: int, func) -> np.ndarray:
    out = np.full(x.shape, np.nan, dtype=float)
    if window <= 0 or x.shape[1] < window:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out[:, window - 1 :] = func(sliding_window_view(x, window, axis=1))
    return out


def _delay(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full(x.shape, np.nan, dtype=float)
    if window < x.shape[1]:
        out[:, window:] = x[:, :-window]
    return out


def _rank_window(v: np.ndarray) -> np.ndarray:
    current = v[..., -1:]
    valid = ~np.isnan(v)
    count = valid.sum(axis=-1)
    result = ((v <= current) & valid).sum(axis=-1) / np.maximum(count, 1)
    return np.where((count > 0) & ~np.isnan(current[..., 0]), result, np.nan)


def _sum(v: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(v).all(axis=-1), np.nan, np.nansum(v, axis=-1))


def _decay(v: np.ndarray) -> np.ndarray:
    weights = np.arange(1, v.shape[-1] + 1, dtype=float)
    valid = ~np.isnan(v)
    denominator = (valid * weights).sum(axis=-1)
    return np.where(denominator > 0, np.nansum(v * weights, axis=-1) / np.maximum(denominator, 1), np.nan)


def _arg(v: np.ndarray, maximum: bool) -> np.ndarray:
    all_nan = np.isnan(v).all(axis=-1)
    masked = np.where(np.isnan(v), -np.inf if maximum else np.inf, v)
    index = masked.argmax(axis=-1) if maximum else masked.argmin(axis=-1)
    return np.where(all_nan, np.nan, v.shape[-1] - 1 - index).astype(float)


def _pair(a: np.ndarray, b: np.ndarray, window: int, correlation: bool) -> np.ndarray:
    def calculate(va: np.ndarray, vb: np.ndarray) -> np.ndarray:
        valid = ~(np.isnan(va) | np.isnan(vb))
        n = valid.sum(axis=-1)
        x, y = np.where(valid, va, 0), np.where(valid, vb, 0)
        nf = np.maximum(n, 1)
        cov = (x * y).sum(axis=-1) / nf - x.sum(axis=-1) * y.sum(axis=-1) / (nf * nf)
        if not correlation:
            return np.where(n >= 2, cov, np.nan)
        vx = (x * x).sum(axis=-1) / nf - (x.sum(axis=-1) / nf) ** 2
        vy = (y * y).sum(axis=-1) / nf - (y.sum(axis=-1) / nf) ** 2
        denominator = np.sqrt(np.maximum(vx, 0) * np.maximum(vy, 0))
        return np.where((n >= 2) & (denominator > _EPS), cov / denominator, np.nan)

    out = np.full(a.shape, np.nan, dtype=float)
    if a.shape[1] >= window:
        out[:, window - 1 :] = calculate(
            sliding_window_view(a, window, axis=1), sliding_window_view(b, window, axis=1)
        )
    return out


def _cs_rank(x: np.ndarray) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    for day in range(x.shape[1]):
        valid = ~np.isnan(x[:, day])
        count = int(valid.sum())
        if count:
            order = np.argsort(x[valid, day], kind="mergesort")
            ranks = np.empty(count)
            ranks[order] = (np.arange(count) + 1) / count
            out[valid, day] = ranks
    return out


def get_op(token: str) -> tuple[object, int] | None:
    parsed = parse_ts_token(token)
    if parsed:
        name, window = parsed
        if name in TS_BINARY_OPS:
            return (lambda a, b, w=window, c=name == "ts_corr": _pair(a, b, w, c)), 2
        functions = {
            "ts_mean": lambda x: _rolling(x, window, lambda v: np.nanmean(v, axis=-1)),
            "ts_std": lambda x: _rolling(x, window, lambda v: np.nanstd(v, axis=-1)),
            "ts_max": lambda x: _rolling(x, window, lambda v: np.nanmax(v, axis=-1)),
            "ts_min": lambda x: _rolling(x, window, lambda v: np.nanmin(v, axis=-1)),
            "ts_delay": lambda x: _delay(x, window),
            "ts_delta": lambda x: x - _delay(x, window),
            "ts_rank": lambda x: _rolling(x, window, _rank_window),
            "ts_sum": lambda x: _rolling(x, window, _sum),
            "decay_linear": lambda x: _rolling(x, window, _decay),
            "ts_argmax": lambda x: _rolling(x, window, lambda v: _arg(v, True)),
            "ts_argmin": lambda x: _rolling(x, window, lambda v: _arg(v, False)),
            "count": lambda x: _rolling(x, window, lambda v: np.where(np.isnan(v).all(-1), np.nan, (v > 0).sum(-1))),
        }
        return functions[name], 1
    if token in CS_OPS:
        return {
            "cs_rank": _cs_rank,
            "cs_zscore": lambda x: (x - np.nanmean(x, 0)) / np.where(np.nanstd(x, 0) < _EPS, np.nan, np.nanstd(x, 0)),
            "cs_demean": lambda x: x - np.nanmean(x, 0),
        }[token], 1
    if token in UNARY_OPS:
        return {
            "neg": lambda x: -x, "abs": np.abs,
            "log": lambda x: np.where(x > 0, np.log(np.where(x > 0, x, 1)), np.nan),
            "sqrt": lambda x: np.where(x >= 0, np.sqrt(np.maximum(x, 0)), np.nan),
            "sign": np.sign, "signedpower": lambda x: np.sign(x) * x * x,
        }[token], 1
    if token in BINARY_OPS:
        return {
            "add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
            "div": lambda a, b: np.where(np.abs(b) < _EPS, np.nan, a / np.where(np.abs(b) < _EPS, 1, b)),
            "gt": lambda a, b: (a > b).astype(float), "lt": lambda a, b: (a < b).astype(float),
            "max": np.maximum, "min": np.minimum,
        }[token], 2
    if token in TERNARY_OPS:
        return (lambda condition, left, right: np.where(condition > 0, left, right)), 3
    return None
