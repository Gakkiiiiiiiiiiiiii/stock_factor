import httpx

from stock_factor.adapters.http.providers import HttpMarketDataProvider
from stock_factor.application.panel import build_feature_panel
from stock_factor.domain.market import MarketDataSnapshot


def test_market_provider_reads_agent_batch_contract(monkeypatch):
    calls = []

    def fake_post(url, json, timeout):
        calls.append((url, json, timeout))
        return httpx.Response(
            200,
            json={
                "contract_version": "market-data.v1",
                "data": {
                    "symbols": ["600000"], "dates": ["2026-08-10"],
                    "bars": {"close": [[10.0]]}, "data_version": "v1",
                    "data_snapshot_id": "s1", "source": "fixture",
                },
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    snapshot = HttpMarketDataProvider("http://market").get_daily_bars(["600000"], "2026-08-10", "2026-08-10")
    assert snapshot.data_snapshot_id == "s1"
    assert calls[0][0] == "http://market/v1/bars/batch"
    assert calls[0][1]["adjust"] == "qfq"


def test_feature_panel_maps_content_v2_fields_at_available_time():
    snapshot = MarketDataSnapshot(
        ["600000"], ["2026-08-10", "2026-08-11"],
        {"open": [[10, 11]], "high": [[11, 12]], "low": [[9, 10]], "close": [[10, 11]], "volume": [[2, 2]], "amount": [[20, 22]], "turnover": [[1, 1]]},
        "v1", "s1", "fixture",
    )
    signals = [
        {"symbol": "600000", "subject_key": "600000", "available_from": "2026-08-10T09:00:00+00:00", "knowledge_kind": "CAUSAL_THESIS", "sentiment": "BULLISH", "truth_status": "EXTERNALLY_VERIFIED", "source_video_id": "v1"},
        {"symbol": "600000", "subject_key": "600000", "available_from": "2026-08-10T09:00:00+00:00", "knowledge_kind": "RISK_CONDITION", "sentiment": "BEARISH", "truth_status": "EXTERNALLY_VERIFIED", "source_video_id": "v2"},
    ]
    panel = build_feature_panel(snapshot, signals)
    assert panel["event_heat"].tolist() == [[0.0, 2.0]]
    assert panel["author_attention_score"].tolist() == [[0.0, 1.0]]
    assert panel["verified_catalyst_count"].tolist() == [[0.0, 1.0]]
    assert panel["verified_risk_count"].tolist() == [[0.0, 1.0]]
    assert panel["cross_video_consensus"].tolist() == [[0.0, 0.0]]
    assert panel["cross_video_disagreement"].tolist() == [[0.0, 1.0]]
