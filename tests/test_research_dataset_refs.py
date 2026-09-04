"""Dataset Ref 与 Candidate Freeze 完整性（详细修改方案 P0-3 / P0-4）。"""

from __future__ import annotations

from stock_factor.domain.content_signal_v5 import FormalContentQuery, FormalContentRef
from stock_factor.domain.datasets import DiscoveryDatasetRef, FinalOosDatasetRef, assert_disjoint
from stock_factor.domain.market_dataset_ref import FormalMarketDatasetRef
from stock_factor.engine.oos_seal import CandidateFreeze


def test_dataset_refs_have_distinct_physical_identity():
    common = {
        "market_snapshot_id": "mds-1",
        "source_data_version": "v1",
        "universe_snapshot_id": "uni-1",
        "feature_schema_version": "features-1",
    }
    discovery = DiscoveryDatasetRef(start="2020-01-01", end="2024-12-31", warmup_start="2019-07-01", **common)
    final_oos = FinalOosDatasetRef(start="2025-01-01", end="2026-06-30", warmup_start="2024-10-01", **common)
    assert discovery.dataset_hash and final_oos.dataset_hash
    assert discovery.dataset_hash != final_oos.dataset_hash
    assert_disjoint(discovery, final_oos)

    # 即使窗口参数相同，scope 不同身份也不同；完全同 scope 同参数则身份一致（内容寻址）。
    same = DiscoveryDatasetRef(start="2020-01-01", end="2024-12-31", warmup_start="2019-07-01", **common)
    assert same.dataset_hash == discovery.dataset_hash


def test_candidate_freeze_carries_full_selection_evidence():
    freeze = CandidateFreeze(
        candidate_hash="cand-1",
        formula=["close", "ts_delay_5", "div"],
        dsl_version="factor-dsl.v1",
        feature_set_version="features-1",
        discovery_snapshot_id="mds-discovery-1",
        final_oos_snapshot_id="mds-oos-1",
        experiment_id="exp-1",
        selection_policy_version="finalist_selection_v1",
        selection_rank=1,
        factor_code_hash="sha:code",
        universe_snapshot_id="uni-1",
        feature_normalization_version="norm-v1",
        selection_config_hash="sha:selcfg",
        discovery_metrics_hash="sha:metrics",
        candidate_count=50,
        code_sha="sha:repo",
        config_hash="sha:config",
    )
    payload = freeze.to_dict()
    # P0-4：freeze 必须能回答"为什么这个因子被选中"
    for field in (
        "candidate_hash",
        "factor_code_hash",
        "discovery_snapshot_id",
        "universe_snapshot_id",
        "feature_set_version",
        "feature_normalization_version",
        "selection_policy_version",
        "selection_config_hash",
        "discovery_metrics_hash",
        "candidate_count",
        "code_sha",
        "config_hash",
        "candidate_frozen_at",
    ):
        assert payload.get(field) is not None, f"freeze 缺少 {field}"
    assert freeze.candidate_rank == 1


def _formal_refs(content_snapshot_id="content-1"):
    market = FormalMarketDatasetRef(
        market_snapshot_id="mds-1",
        manifest_hash="market-manifest",
        calendar_version="cal-v1",
        universe_version="uni-v1",
        corporate_action_version="ca-v1",
        tradability_version="trad-v1",
        available_from="2026-08-01T00:00:00+00:00",
        start="2026-08-01",
        end="2026-08-31",
    )
    query = FormalContentQuery(
        contract="content-factor-signal.v5.1",
        checksum="sha256:content",
        content_snapshot_id=content_snapshot_id,
        business_as_of="2026-08-10T00:00:00+00:00",
        knowledge_as_of="2026-08-10T00:00:00+00:00",
        availability_as_of="2026-08-10T00:00:00+00:00",
        pit_mode="PUBLIC_STRICT",
        signal_policy_version="policy-v1",
        min_support=2,
        producer_commit="content@abc123",
    )
    return market, FormalContentRef.from_query(query, "content-manifest")


def test_formal_dataset_identity_requires_and_binds_content_ref():
    market, content = _formal_refs()
    common = dict(
        market_snapshot_id="mds-1",
        source_data_version="v1",
        universe_snapshot_id="uni-1",
        feature_schema_version="features-1",
        start="2026-08-01",
        end="2026-08-10",
        formal_market_ref=market,
    )
    missing_content = DiscoveryDatasetRef(**common)
    assert missing_content.formal_eligible is False
    formal = DiscoveryDatasetRef(**common, formal_content_ref=content)
    changed = DiscoveryDatasetRef(**common, formal_content_ref=_formal_refs("content-2")[1])
    assert formal.formal_eligible is True
    assert formal.content_ref_hash == content.ref_hash
    assert formal.dataset_hash != changed.dataset_hash
    assert formal.dataset_hash == DiscoveryDatasetRef(**common, formal_content_ref=content).dataset_hash
