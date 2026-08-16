import httpx

from stock_factor.adapters.http.providers import HttpMarketDataProvider
from stock_factor.application.panel import build_feature_panel
from stock_factor.domain.market import MarketDataSnapshot


def test_mining_job_rejected_when_quant_unavailable(tmp_path):
    """§90：Quant 市场数据不可用 => 禁止启动新 Mining（503 DATA_NOT_READY）。"""
    from fastapi.testclient import TestClient

    from stock_factor.api.dependencies import build_application
    from stock_factor.api.main import create_app

    from tests.test_integration import FixtureContent

    class BrokenMarket:
        def get_daily_bars(self, symbols, start, end, adjust):
            raise httpx.ConnectError("quant unreachable")

    application = build_application(f"sqlite:///{tmp_path / 'degrade.db'}", BrokenMarket(), FixtureContent())
    client = TestClient(create_app(application))
    response = client.post("/api/v1/mining/jobs", json={"symbols": ["600000"]})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "DATA_NOT_READY"


def test_market_provider_reads_agent_batch_contract(monkeypatch):
    calls = []

    def fake_post(url, json, timeout, headers=None):
        calls.append((url, json, timeout, headers))
        return httpx.Response(
            200,
            json={
                "contract_version": "market-data.v1",
                "data": {
                    "symbols": ["600000"],
                    "dates": ["2026-08-10"],
                    "bars": {"close": [[10.0]]},
                    "data_version": "v1",
                    "data_snapshot_id": "s1",
                    "source": "fixture",
                },
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    snapshot = HttpMarketDataProvider("http://market").get_daily_bars(["600000"], "2026-08-10", "2026-08-10")
    assert snapshot.data_snapshot_id == "s1"
    # §12/§76：事实源切换为 quant 后，主路径为 /api/v1/market/bars/batch。
    assert calls[0][0] == "http://market/api/v1/market/bars/batch"
    assert calls[0][1]["adjust"] == "qfq"
    # §32：出站调用透传 Trace Headers。
    assert calls[0][3]["X-Caller-Service"] == "stock_factor"
    assert calls[0][3]["X-Trace-Id"]


def test_market_provider_defaults_to_quant(monkeypatch):
    monkeypatch.delenv("MARKET_DATA_SERVICE_URL", raising=False)
    provider = HttpMarketDataProvider()
    assert provider._url == "http://quant:8011"  # noqa: SLF001


def test_market_provider_falls_back_to_legacy_path(monkeypatch):
    """迁移期兼容：旧 market-data-service 只提供 /v1/bars/batch（§12）。"""
    calls = []

    def fake_post(url, json, timeout, headers=None):
        calls.append(url)
        if url.endswith("/api/v1/market/bars/batch"):
            return httpx.Response(404, request=httpx.Request("POST", url))
        return httpx.Response(
            200,
            json={
                "contract_version": "market-data.v1",
                "data": {
                    "symbols": ["600000"],
                    "dates": ["2026-08-10"],
                    "bars": {"close": [[10.0]]},
                    "data_version": "v1",
                    "data_snapshot_id": "s1",
                    "source": "fixture",
                },
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    snapshot = HttpMarketDataProvider("http://legacy-market").get_daily_bars(["600000"], "2026-08-10", "2026-08-10")
    assert snapshot.data_snapshot_id == "s1"
    assert calls == [
        "http://legacy-market/api/v1/market/bars/batch",
        "http://legacy-market/v1/bars/batch",
    ]


def test_feature_panel_maps_content_v2_fields_at_available_time():
    snapshot = MarketDataSnapshot(
        ["600000"],
        ["2026-08-10", "2026-08-11"],
        {
            "open": [[10, 11]],
            "high": [[11, 12]],
            "low": [[9, 10]],
            "close": [[10, 11]],
            "volume": [[2, 2]],
            "amount": [[20, 22]],
            "turnover": [[1, 1]],
        },
        "v1",
        "s1",
        "fixture",
    )
    signals = [
        {
            "symbol": "600000",
            "subject_key": "600000",
            "available_from": "2026-08-10T09:00:00+00:00",
            "knowledge_kind": "CAUSAL_THESIS",
            "sentiment": "BULLISH",
            "truth_status": "EXTERNALLY_VERIFIED",
            "source_video_id": "v1",
        },
        {
            "symbol": "600000",
            "subject_key": "600000",
            "available_from": "2026-08-10T09:00:00+00:00",
            "knowledge_kind": "RISK_CONDITION",
            "sentiment": "BEARISH",
            "truth_status": "EXTERNALLY_VERIFIED",
            "source_video_id": "v2",
        },
    ]
    panel = build_feature_panel(snapshot, signals)
    assert panel["event_heat"].tolist() == [[0.0, 2.0]]
    assert panel["author_attention_score"].tolist() == [[0.0, 1.0]]
    assert panel["verified_catalyst_count"].tolist() == [[0.0, 1.0]]
    assert panel["verified_risk_count"].tolist() == [[0.0, 1.0]]
    assert panel["cross_video_consensus"].tolist() == [[0.0, 0.0]]
    assert panel["cross_video_disagreement"].tolist() == [[0.0, 1.0]]
