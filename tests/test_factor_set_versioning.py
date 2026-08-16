"""FactorSet 版本化（详细修改方案 §11 / §17）。"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from stock_factor.adapters.postgres.models import Base
from stock_factor.application.factor_set_service import (
    FactorSetService,
    InMemoryFactorSetStore,
    PostgresFactorSetStore,
)
from stock_factor.domain.factor_set import FactorSet, factor_set_from_factors


def test_factor_set_content_addressed_identity():
    factors = [{"factor_id": "f-b", "version": 1}, {"factor_id": "f-a", "version": 2}]
    first = factor_set_from_factors(factors)
    second = factor_set_from_factors(list(reversed(factors)))
    assert first.factor_set_id == second.factor_set_id  # 与顺序无关
    assert first.factor_set_version.startswith("factor-set-")
    assert first.weights and abs(sum(first.weights) - 1.0) < 1e-6
    assert first.factor_ids == ("f-a", "f-b")
    assert first.factor_versions == (2, 1)


def test_factor_set_length_mismatch_rejected():
    with pytest.raises(ValueError):
        FactorSet(factor_ids=("f-1",), factor_versions=(1, 2), weights=(1.0,))


def test_service_supersedes_previous_active():
    service = FactorSetService(InMemoryFactorSetStore())
    v1 = service.publish_from_factors([{"factor_id": "f-1", "version": 1}])
    v2 = service.publish_from_factors([{"factor_id": "f-1", "version": 1}, {"factor_id": "f-2", "version": 1}])
    assert v1.factor_set_id != v2.factor_set_id
    assert service.get(v1.factor_set_id).status == "SUPERSEDED"
    assert service.current().factor_set_id == v2.factor_set_id
    # 幂等：相同内容再次发布不产生新版本
    again = service.publish_from_factors([{"factor_id": "f-1", "version": 1}, {"factor_id": "f-2", "version": 1}])
    assert again.factor_set_id == v2.factor_set_id


def test_postgres_store_roundtrip(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'fs.db').as_posix()}")
    Base.metadata.create_all(engine)
    store = PostgresFactorSetStore(sessionmaker(engine, expire_on_commit=False))
    service = FactorSetService(store)
    published = service.publish_from_factors(
        [{"factor_id": "f-1", "version": 3}, {"factor_id": "f-2", "version": 1}],
        research_experiment_ids=("exp-1",),
    )
    loaded = service.get(published.factor_set_id)
    assert loaded is not None
    assert loaded.factor_ids == published.factor_ids
    assert loaded.factor_versions == published.factor_versions
    assert loaded.research_experiment_ids == ("exp-1",)
    assert service.current().factor_set_id == published.factor_set_id
