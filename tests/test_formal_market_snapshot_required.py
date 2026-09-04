from datetime import UTC, datetime

import pytest

from stock_factor.application.market_snapshot_validator import validate_formal_market_snapshot
from stock_factor.application.mining import FactorMiningService
from stock_factor.domain.market_dataset_ref import FormalMarketDatasetRef


def _ref(**overrides):
    values = {
        "market_snapshot_id": "snap-1",
        "manifest_hash": "manifest-1",
        "calendar_version": "cal-1",
        "universe_version": "uni-1",
        "corporate_action_version": "ca-1",
        "tradability_version": "trad-1",
        "quality_seal": "PASS",
        "available_from": "2026-08-09T00:00:00+00:00",
        "start": "2026-08-10",
        "end": "2026-08-11",
    }
    values.update(overrides)
    return FormalMarketDatasetRef(**values)


def test_formal_ref_has_canonical_hash_and_rejects_invalid_identity():
    ref = _ref()
    assert len(ref.ref_hash) == 64
    with pytest.raises(ValueError):
        _ref(manifest_hash="")
    with pytest.raises(ValueError):
        _ref(quality_seal="UNKNOWN")
    with pytest.raises(ValueError):
        _ref(available_from="2026-08-09T00:00:00")
    with pytest.raises(ValueError):
        _ref(start="2026-08-12", end="2026-08-11")


def test_validator_checks_range_cutoff_and_hash():
    ref = _ref()
    assert (
        validate_formal_market_snapshot(
            ref,
            requested_start="2026-08-10",
            requested_end="2026-08-11",
            as_of=datetime(2026, 8, 12, tzinfo=UTC),
            expected_manifest_hash="manifest-1",
            expected_ref_hash=ref.ref_hash,
        )
        is ref
    )
    with pytest.raises(ValueError):
        validate_formal_market_snapshot(ref, requested_start="2026-08-09", requested_end="2026-08-11")
    with pytest.raises(ValueError):
        validate_formal_market_snapshot(
            ref,
            requested_start="2026-08-10",
            requested_end="2026-08-11",
            as_of=datetime(2026, 8, 8, tzinfo=UTC),
        )
    with pytest.raises(ValueError):
        validate_formal_market_snapshot(
            ref, requested_start="2026-08-10", requested_end="2026-08-11", expected_ref_hash="wrong"
        )


def test_equivalent_instants_have_the_same_canonical_hash():
    assert (
        _ref(available_from="2026-08-09T00:00:00+00:00").ref_hash
        == _ref(available_from="2026-08-09T08:00:00+08:00").ref_hash
    )


def test_unknown_research_mode_is_rejected_before_market_access():
    with pytest.raises(ValueError, match="research_mode"):
        FactorMiningService(None, None, None).run({"symbols": ["600000"], "research_mode": "TYPO"})
