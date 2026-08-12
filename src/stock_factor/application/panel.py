from __future__ import annotations

import numpy as np

from stock_factor.domain.market import MarketDataSnapshot


def build_feature_panel(snapshot: MarketDataSnapshot, signals: list[dict]) -> dict[str, np.ndarray]:
    symbols, dates = snapshot.symbols, snapshot.dates
    shape = (len(symbols), len(dates))
    panel = {name: np.asarray(snapshot.bars.get(name, np.full(shape, np.nan)), dtype=float) for name in ("open", "high", "low", "close", "volume", "amount", "turnover")}
    panel["vwap"] = np.where(panel["volume"] != 0, panel["amount"] / panel["volume"], np.nan)
    panel["ret"] = np.full(shape, np.nan)
    panel["ret"][:, 1:] = panel["close"][:, 1:] / panel["close"][:, :-1] - 1
    content_names = ("event_heat", "theme_sentiment", "video_bullish_claim_count", "video_bearish_claim_count", "verified_catalyst_count", "verified_risk_count", "author_attention_score", "cross_video_consensus", "cross_video_disagreement")
    panel.update({name: np.zeros(shape) for name in content_names})
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    for signal in signals:
        symbol = signal.get("symbol")
        if symbol not in symbol_index:
            continue
        available = str(signal.get("available_from", ""))[:10]
        day = next((index for index, value in enumerate(dates) if value >= available), None)
        if day is None:
            continue
        row, confidence = symbol_index[symbol], float(signal.get("confidence", 0))
        sentiment = signal.get("sentiment", "NEUTRAL")
        panel["event_heat"][row, day] += confidence
        panel["theme_sentiment"][row, day] += confidence * (1 if sentiment == "BULLISH" else -1 if sentiment == "BEARISH" else 0)
        panel["video_bullish_claim_count"][row, day] += sentiment == "BULLISH"
        panel["video_bearish_claim_count"][row, day] += sentiment == "BEARISH"
        panel["verified_catalyst_count"][row, day] += signal.get("kind") == "CATALYST"
        panel["verified_risk_count"][row, day] += signal.get("kind") == "RISK"
    return panel
