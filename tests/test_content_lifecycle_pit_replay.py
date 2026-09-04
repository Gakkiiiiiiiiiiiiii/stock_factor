import pytest

from stock_factor.application.content_signal_eligibility import evaluate_content_eligibility
from stock_factor.domain.content_signal_v5 import ContentSignalV5, FormalContentQuery, FormalContentRef


def query(**overrides):
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
    values.update(overrides)
    return FormalContentQuery(**values)


def signal(**overrides):
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
    values.update(overrides)
    return ContentSignalV5(**values)


def test_content_ref_hash_is_stable_for_equivalent_clock_instants():
    first = query()
    second = query(
        business_as_of="2026-08-10T08:00:00+08:00",
        knowledge_as_of="2026-08-10T08:00:00+08:00",
        availability_as_of="2026-08-10T08:00:00+08:00",
    )
    assert (
        FormalContentRef.from_query(first, "manifest-1").ref_hash
        == FormalContentRef.from_query(second, "manifest-1").ref_hash
    )


def test_content_eligibility_rejects_unknown_proxy_and_late_lifecycle():
    q = query()
    result = evaluate_content_eligibility(
        [
            signal(source_availability_quality="UNKNOWN"),
            signal(signal_id="sig-2", source_availability_quality="PROXY"),
            signal(signal_id="sig-3", lifecycle_as_of="2026-08-11T00:00:00+00:00"),
        ],
        q,
    )
    assert not result.formal_eligible
    assert len(result.accepted) == 0
    assert len(result.rejected) == 3
    proxy = evaluate_content_eligibility([signal(source_availability_quality="PROXY")], q, allow_proxy=True)
    assert proxy.formal_eligible
    query_proxy = query(allow_proxy=True)
    assert evaluate_content_eligibility([signal(source_availability_quality="PROXY")], query_proxy).formal_eligible
    assert not evaluate_content_eligibility([signal(truth_status="UNCONFIRMED")], q).formal_eligible
    assert not evaluate_content_eligibility([signal(producer_commit="content@other")], q).formal_eligible


def test_content_signal_rejects_naive_or_extra_temporal_payload():
    with pytest.raises(ValueError):
        signal(available_from="2026-08-09T00:00:00")
    with pytest.raises(ValueError):
        signal(unexpected="today")
