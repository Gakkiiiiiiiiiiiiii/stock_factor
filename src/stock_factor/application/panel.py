from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta, timezone

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
    "content_attention_score",
    "author_attention_score",
    "cross_video_consensus",
    "cross_video_disagreement",
)


_CN_TZ = timezone(timedelta(hours=8))
_FEATURE_CUTOFF = time(9, 25)


def _available_from(signal: dict) -> datetime | None:
    # ``as_of`` is an observation timestamp, not visibility permission, so it
    # must never be used as a fallback for the anti-lookahead boundary.
    raw = signal.get("available_from")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).astimezone(_CN_TZ)
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
        available_from = _available_from(signal)
        if available_from is None:
            continue
        rows = [code_index[symbol] for symbol in _symbols(signal) if symbol in code_index]
        for day_index, current_day in enumerate(day_dates):
            cutoff = datetime.combine(current_day, _FEATURE_CUTOFF, tzinfo=_CN_TZ)
            delta = (current_day - available_from.date()).days
            if available_from > cutoff or delta < 0 or delta > lookback_days:
                continue
            for row in rows:
                cells.setdefault((row, day_index), []).append(signal)

    for (row, day_index), cell_signals in cells.items():
        videos = {signal.get("source_video_id") for signal in cell_signals if signal.get("source_video_id")}
        video_sentiments: dict[str, set[str]] = {}
        subject_video_sentiments: dict[str, dict[str, set[str]]] = {}
        attentions: list[float] = []
        consensuses: list[float] = []
        disagreements: list[float] = []
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
            if signal.get("content_attention_score") is not None:
                attentions.append(float(signal["content_attention_score"]))
            if signal.get("cross_video_consensus") is not None:
                consensuses.append(float(signal["cross_video_consensus"]))
            if signal.get("cross_video_disagreement") is not None:
                disagreements.append(float(signal["cross_video_disagreement"]))
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
        # Content owns attention and corroboration semantics.  The legacy
        # author_attention token stays as a compatibility alias for formulas.
        # Older stored signals did not expose the canonical semantic fields;
        # retain their historical calculation solely as a read-compatibility
        # fallback. New Content v2 signals always take the branch above.
        attention = float(np.mean(attentions)) if attentions else len(cell_signals) / max(len(videos), 1)
        panels["content_attention_score"][row, day_index] = attention
        panels["author_attention_score"][row, day_index] = attention
        if consensuses or disagreements:
            panels["cross_video_consensus"][row, day_index] = float(np.mean(consensuses)) if consensuses else 0.0
            panels["cross_video_disagreement"][row, day_index] = float(np.mean(disagreements)) if disagreements else 0.0
        else:
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
