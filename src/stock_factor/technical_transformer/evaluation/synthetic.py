from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_pattern(name: str, rows: int = 240, seed: int = 7) -> pd.DataFrame:
    """Generate deterministic OHLCV fixtures for label-generator sanity tests."""
    rng = np.random.default_rng(seed)
    name = name.lower()
    close = np.full(rows, 100.0)
    if name == "linear_uptrend":
        close = np.linspace(80, 130, rows)
    elif name == "linear_downtrend":
        close = np.linspace(130, 80, rows)
    elif name == "range":
        close = 100 + 4 * np.sin(np.arange(rows) / 8)
    elif name == "boll_squeeze":
        close = 100 + np.r_[np.sin(np.arange(rows - 30) / 7), np.sin(np.arange(30) / 7) * 0.15]
    elif name == "boll_expansion":
        close = 100 + np.r_[np.sin(np.arange(rows - 30) / 7) * 0.15, np.sin(np.arange(30) / 3) * 5]
    elif name == "spring":
        close = 100 + 2 * np.sin(np.arange(rows) / 9)
        close[-4] = 94; close[-3] = 99; close[-2] = 102; close[-1] = 103
    elif name == "upthrust":
        close = 100 + 2 * np.sin(np.arange(rows) / 9)
        close[-4] = 106; close[-3] = 101; close[-2] = 98; close[-1] = 97
    else:
        raise ValueError(f"unknown synthetic pattern: {name}")
    volume = np.full(rows, 100_000.0)
    if name in {"spring", "upthrust", "boll_expansion"}:
        volume[-4:] *= 2.5
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    return pd.DataFrame({
        "trading_date": dates, "symbol": "SYN.SZ", "open": close - 0.5, "high": close + 1.0,
        "low": close - 1.0, "close": close, "volume": volume, "amount": volume * close,
        "turnover": 0.01, "listing_days": np.arange(1, rows + 1),
    })
