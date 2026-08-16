"""Dataset Ref 与 Candidate Freeze 完整性（详细修改方案 P0-3 / P0-4）。"""
from __future__ import annotations

from stock_factor.domain.datasets import DiscoveryDatasetRef, FinalOosDatasetRef, assert_disjoint
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
        "candidate_hash", "factor_code_hash", "discovery_snapshot_id", "universe_snapshot_id",
        "feature_set_version", "feature_normalization_version", "selection_policy_version",
        "selection_config_hash", "discovery_metrics_hash", "candidate_count",
        "code_sha", "config_hash", "candidate_frozen_at",
    ):
        assert payload.get(field) is not None, f"freeze 缺少 {field}"
    assert freeze.candidate_rank == 1
