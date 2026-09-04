import httpx
import pytest

from stock_factor.adapters.http.providers import ExploratoryMarketDataProvider, HttpMarketDataProvider
from stock_factor.domain.market_dataset_ref import FormalMarketDatasetRef


def _ref():
    return FormalMarketDatasetRef(
        market_snapshot_id="snap-1",
        manifest_hash="manifest-1",
        calendar_version="cal-1",
        universe_version="uni-1",
        corporate_action_version="ca-1",
        tradability_version="trad-1",
        available_from="2026-08-09T00:00:00+00:00",
        start="2026-08-10",
        end="2026-08-10",
    )


def test_formal_provider_does_not_fallback_after_404(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append(url)
        return httpx.Response(404, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(httpx.HTTPStatusError):
        HttpMarketDataProvider("http://quant").get_daily_bars(
            ["600000"], "2026-08-10", "2026-08-10", formal_market_ref=_ref()
        )
    assert calls == ["http://quant/api/v1/market/bars/batch"]


def test_formal_provider_does_not_fallback_after_network_error(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append(url)
        raise httpx.ConnectError("quant unavailable")

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(httpx.ConnectError):
        HttpMarketDataProvider("http://quant").get_daily_bars(
            ["600000"], "2026-08-10", "2026-08-10", formal_market_ref=_ref()
        )
    assert calls == ["http://quant/api/v1/market/bars/batch"]


def test_formal_provider_requires_exact_contract_and_manifest(monkeypatch):
    def fake_post(url, **kwargs):
        return httpx.Response(
            200,
            json={"contract_version": "market-data.v1", "data": {}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(ValueError, match="formal market contract"):
        HttpMarketDataProvider("http://quant").get_daily_bars(
            ["600000"], "2026-08-10", "2026-08-10", formal_market_ref=_ref()
        )


def test_exploratory_provider_is_explicitly_not_formal(monkeypatch):
    def fake_post(url, **kwargs):
        return httpx.Response(
            200,
            json={
                "contract_version": "market-data.v1",
                "data": {
                    "symbols": ["600000"],
                    "dates": ["2026-08-10"],
                    "bars": {"close": [[10.0]]},
                    "data_version": "v1",
                    "data_snapshot_id": "snap-1",
                },
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    snapshot = ExploratoryMarketDataProvider("http://legacy").get_daily_bars(["600000"], "2026-08-10", "2026-08-10")
    assert snapshot.formal_eligible is False
