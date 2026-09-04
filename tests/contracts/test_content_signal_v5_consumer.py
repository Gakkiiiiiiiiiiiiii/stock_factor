import httpx
import pytest

from stock_factor.adapters.http.content_v5_provider import FormalContentSignalProviderV5
from stock_factor.domain.content_signal_v5 import ContentSignalV5, FormalContentQuery, FormalContentRef


def test_formal_content_provider_rejects_old_contract(monkeypatch):
    query = FormalContentQuery(
        contract="content-factor-signal.v5.1",
        checksum="sha256:content",
        content_snapshot_id="content-1",
        business_as_of="2026-08-10T00:00:00+00:00",
        knowledge_as_of="2026-08-10T00:00:00+00:00",
        availability_as_of="2026-08-10T00:00:00+00:00",
        pit_mode="PUBLIC_STRICT",
        signal_policy_version="policy-v1",
        min_support=2,
        producer_commit="content@abc123",
    )

    def fake_post(url, **kwargs):
        return httpx.Response(
            200, json={"contract_version": "content-factor-signal.v3"}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    expected = FormalContentRef.from_query(query, "manifest-1")
    with pytest.raises(ValueError, match="unsupported"):
        FormalContentSignalProviderV5("http://content").load_signals(
            ["600000"], "2026-08-10", "2026-08-10", query=query, expected_ref=expected
        )


def test_formal_content_provider_binds_query_identity_and_checksum(monkeypatch):
    query = FormalContentQuery(
        contract="content-factor-signal.v5.1",
        checksum="sha256:content",
        content_snapshot_id="content-1",
        business_as_of="2026-08-10T00:00:00+00:00",
        knowledge_as_of="2026-08-10T00:00:00+00:00",
        availability_as_of="2026-08-10T00:00:00+00:00",
        pit_mode="PUBLIC_STRICT",
        signal_policy_version="policy-v1",
        min_support=2,
        producer_commit="content@abc123",
    )
    expected = FormalContentRef.from_query(query, "manifest-1")
    calls = []

    def fake_post(url, **kwargs):
        calls.append(kwargs["json"])
        return httpx.Response(
            200,
            json={
                "contract_version": query.contract,
                "data": {
                    **query.model_dump(mode="json"),
                    "manifest_hash": expected.manifest_hash,
                    "items": [],
                },
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = FormalContentSignalProviderV5("http://content").load_signals(
        ["600000"], "2026-08-10", "2026-08-10", query=query, expected_ref=expected
    )
    assert result == []
    assert calls[0]["content_snapshot_id"] == "content-1"
    assert calls[0]["availability_as_of"] == "2026-08-10T00:00:00Z"
    assert calls[0]["request_id"]
    assert calls[0]["request_id"] not in expected.ref_hash


def test_formal_content_provider_requires_bound_ref_and_rejects_response_extras(monkeypatch):
    query = FormalContentQuery(
        contract="content-factor-signal.v5.1",
        checksum="sha256:content",
        content_snapshot_id="content-1",
        business_as_of="2026-08-10T00:00:00+00:00",
        knowledge_as_of="2026-08-10T00:00:00+00:00",
        availability_as_of="2026-08-10T00:00:00+00:00",
        pit_mode="PUBLIC_STRICT",
        signal_policy_version="policy-v1",
        min_support=2,
        producer_commit="content@abc123",
    )
    expected = FormalContentRef.from_query(query, "manifest-1")

    def fake_post(url, **kwargs):
        return httpx.Response(
            200,
            json={
                "contract_version": query.contract,
                "data": {
                    **query.model_dump(mode="json"),
                    "manifest_hash": expected.manifest_hash,
                    "items": [],
                    "unexpected": True,
                },
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = FormalContentSignalProviderV5("http://content")
    with pytest.raises(ValueError, match="expected_ref"):
        provider.load_signals(["600000"], "2026-08-10", "2026-08-10", query=query)
    with pytest.raises(ValueError, match="unexpected"):
        provider.load_signals(["600000"], "2026-08-10", "2026-08-10", query=query, expected_ref=expected)


def test_content_v5_rejects_unknown_quality_and_legacy_support_type():
    query = FormalContentQuery(
        contract="content-factor-signal.v5.1",
        checksum="sha256:content",
        content_snapshot_id="content-1",
        business_as_of="2026-08-10T00:00:00+00:00",
        knowledge_as_of="2026-08-10T00:00:00+00:00",
        availability_as_of="2026-08-10T00:00:00+00:00",
        pit_mode="PUBLIC_STRICT",
        signal_policy_version="policy-v1",
        min_support=2,
        producer_commit="content@abc123",
    )
    with pytest.raises(ValueError):
        FormalContentQuery(**{**query.model_dump(), "min_support": "SOURCE_SUPPORTED"})
    with pytest.raises(ValueError):
        ContentSignalV5(
            signal_id="sig-1",
            symbol="600000",
            subject_key="600000",
            occurrence_at="2026-08-09T00:00:00+00:00",
            business_as_of="2026-08-09T00:00:00+00:00",
            knowledge_projection_at="2026-08-09T00:00:00+00:00",
            source_available_at="2026-08-09T00:00:00+00:00",
            available_from="2026-08-09T00:00:00+00:00",
            source_availability_quality="UNTRUSTED",
            lifecycle_as_of="2026-08-09T00:00:00+00:00",
            sentiment="BULLISH",
            knowledge_kind="FACT",
            truth_status="EXTERNALLY_VERIFIED",
            source_video_id="video-1",
            support_count=2,
            producer_commit="content@abc123",
        )


@pytest.mark.parametrize(
    "missing",
    [
        "contract",
        "checksum",
        "content_snapshot_id",
        "business_as_of",
        "knowledge_as_of",
        "availability_as_of",
        "pit_mode",
        "signal_policy_version",
        "min_support",
        "producer_commit",
    ],
)
def test_formal_query_missing_identity_field_is_rejected(missing):
    values = {
        "contract": "content-factor-signal.v5.1",
        "checksum": "sha256:content",
        "content_snapshot_id": "content-1",
        "business_as_of": "2026-08-10T00:00:00+00:00",
        "knowledge_as_of": "2026-08-10T00:00:00+00:00",
        "availability_as_of": "2026-08-10T00:00:00+00:00",
        "pit_mode": "PUBLIC_STRICT",
        "signal_policy_version": "policy-v1",
        "min_support": 2,
        "producer_commit": "content@abc123",
    }
    values.pop(missing)
    with pytest.raises(ValueError):
        FormalContentQuery(**values)


@pytest.mark.parametrize(
    "missing",
    ["occurrence_at", "available_from", "source_available_at", "lifecycle_as_of", "support_count", "producer_commit"],
)
def test_formal_signal_missing_temporal_or_lineage_field_is_rejected(missing):
    values = {
        "signal_id": "sig-1",
        "symbol": "600000",
        "subject_key": "600000",
        "occurrence_at": "2026-08-09T00:00:00+00:00",
        "business_as_of": "2026-08-09T00:00:00+00:00",
        "knowledge_projection_at": "2026-08-09T00:00:00+00:00",
        "source_available_at": "2026-08-09T00:00:00+00:00",
        "available_from": "2026-08-09T00:00:00+00:00",
        "source_availability_quality": "VERIFIED",
        "lifecycle_as_of": "2026-08-09T00:00:00+00:00",
        "sentiment": "BULLISH",
        "knowledge_kind": "FACT",
        "truth_status": "EXTERNALLY_VERIFIED",
        "source_video_id": "video-1",
        "support_count": 2,
        "producer_commit": "content@abc123",
    }
    values.pop(missing)
    with pytest.raises(ValueError):
        ContentSignalV5(**values)
