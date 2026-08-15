from __future__ import annotations

from datetime import date, datetime

import numpy as np

from stock_factor.domain.market import MarketDataSnapshot
from stock_factor.ports.symbols import normalize_symbol

_BULLISH = "BULLISH"
_BEARISH = "BEARISH"
_VERIFIED = "EXTERNALLY_VERIFIED"
_CATALYST_KINDS = frozenset({"FACT", "POLICY_FACT", "FINANCIAL_METRIC", "CAUSAL_THESIS"})
_RISK_KINDS = frozenset({"RISK_CONDITION"})
_CONTENT_NAMES = (
    "event_heat",
    "theme_sentiment",
    "video_bullish_claim_count",
    "video_bearish_claim_count",
    "verified_catalyst_count",
    "verified_risk_count",
    "author_attention_score",
    "cross_video_consensus",
    "cross_video_disagreement",
)


def _effective_date(signal: dict) -> date | None:
    # A content claim can only enter a factor panel after it was available to
    # the trading system.  ``as_of`` is an observation timestamp, not a
    # visibility permission, so it must never be used as a fallback here.
    raw = signal.get("available_from")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _symbols(signal: dict) -> set[str]:
    values = (signal.get("symbol"), signal.get("subject_key"), signal.get("subject"))
    resolved: set[str] = set()
    for value in values:
        if not value:
            continue
        try:
            resolved.add(normalize_symbol(str(value)))
        except ValueError:
            continue
    return resolved


def _content_feature_panel(
    symbols: list[str], dates: list[str], signals: list[dict], lookback_days: int = 5
) -> dict[str, np.ndarray]:
    """Port of the baseline KnowledgeUnit V2 feature rules.

    Each unit becomes visible on the first trading day after its
    ``available_from`` timestamp
    and remains in a bounded lookback window. This preserves the baseline's
    anti-lookahead contract while keeping the Factor service HTTP-only.
    """

    shape = (len(symbols), len(dates))
    panels = {name: np.zeros(shape) for name in _CONTENT_NAMES}
    code_index = {normalize_symbol(symbol): index for index, symbol in enumerate(symbols)}
    day_dates = [date.fromisoformat(str(value)[:10]) for value in dates]
    cells: dict[tuple[int, int], list[dict]] = {}
    for signal in signals:
        if str(signal.get("review_status") or "").upper() == "REJECTED":
            continue
        effective = _effective_date(signal)
        if effective is None:
            continue
        rows = [code_index[symbol] for symbol in _symbols(signal) if symbol in code_index]
        for day_index, current_day in enumerate(day_dates):
            delta = (current_day - effective).days
            if delta <= 0 or delta > lookback_days:
                continue
            for row in rows:
                cells.setdefault((row, day_index), []).append(signal)

    for (row, day_index), cell_signals in cells.items():
        videos = {signal.get("source_video_id") for signal in cell_signals if signal.get("source_video_id")}
        video_sentiments: dict[str, set[str]] = {}
        subject_video_sentiments: dict[str, dict[str, set[str]]] = {}
        for signal in cell_signals:
            sentiment = str(signal.get("sentiment") or "").upper()
            kind = str(signal.get("knowledge_kind") or signal.get("kind") or "").upper()
            verified = str(signal.get("truth_status") or "").upper() == _VERIFIED
            video_id = str(
                signal.get("source_video_id") or signal.get("knowledge_uid") or signal.get("signal_id") or ""
            )
            if sentiment in {_BULLISH, _BEARISH}:
                video_sentiments.setdefault(video_id, set()).add(sentiment)
                subject = str(signal.get("subject_key") or signal.get("subject") or "")
                if subject:
                    subject_video_sentiments.setdefault(subject, {}).setdefault(video_id, set()).add(sentiment)
            panels["video_bullish_claim_count"][row, day_index] += sentiment == _BULLISH
            panels["video_bearish_claim_count"][row, day_index] += sentiment == _BEARISH
            panels["verified_catalyst_count"][row, day_index] += (
                verified and sentiment == _BULLISH and kind in _CATALYST_KINDS
            )
            panels["verified_risk_count"][row, day_index] += verified and (kind in _RISK_KINDS or sentiment == _BEARISH)

        panels["event_heat"][row, day_index] = len(videos)
        panels["theme_sentiment"][row, day_index] = sum(
            1 if directions == {_BULLISH} else -1 if directions == {_BEARISH} else 0
            for directions in video_sentiments.values()
        )
        panels["author_attention_score"][row, day_index] = len(cell_signals) / max(len(videos), 1)
        for per_video in subject_video_sentiments.values():
            if len(per_video) < 2:
                continue
            directions = set().union(*per_video.values())
            if directions == {_BULLISH}:
                panels["cross_video_consensus"][row, day_index] += 1
            elif directions == {_BEARISH}:
                panels["cross_video_consensus"][row, day_index] -= 1
            else:
                panels["cross_video_disagreement"][row, day_index] += 1
    return panels


def build_feature_panel(snapshot: MarketDataSnapshot, signals: list[dict]) -> dict[str, np.ndarray]:
    symbols, dates = snapshot.symbols, snapshot.dates
    shape = (len(symbols), len(dates))
    panel = {
        name: np.asarray(snapshot.bars.get(name, np.full(shape, np.nan)), dtype=float)
        for name in ("open", "high", "low", "close", "volume", "amount", "turnover")
    }
    panel["vwap"] = np.where(panel["volume"] != 0, panel["amount"] / panel["volume"], np.nan)
    panel["ret"] = np.full(shape, np.nan)
    panel["ret"][:, 1:] = panel["close"][:, 1:] / panel["close"][:, :-1] - 1
    panel.update(_content_feature_panel(symbols, dates, signals))
    return panel
