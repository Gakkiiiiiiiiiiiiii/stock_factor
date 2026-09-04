from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import httpx
import jsonschema
import pytest
import yaml

from scripts.contracts.diff_schema import semantic_diff
from scripts.contracts.verify_manifest import ManifestVerificationError, verify_manifest
from stock_factor.adapters.http.content_v5_provider import FormalContentSignalProviderV5
from stock_factor.adapters.http.providers import _decode_market_response
from stock_factor.domain.content_signal_v5 import FormalContentQuery, FormalContentRef
from stock_factor.domain.market_dataset_ref import FormalMarketDatasetRef
from stock_factor.domain.research_artifact import ResearchArtifactV2

ROOT = Path(__file__).parents[3]
FIXTURES = Path(__file__).parent / "fixtures"


def test_formal_manifest_inventory_is_complete_and_checksum_bound():
    report = verify_manifest(ROOT / "contracts/platform-manifest.yaml", today=date(2026, 9, 4))
    entries = {entry["name"]: entry for entry in report["contracts"]}
    assert report["valid"] is True
    assert {
        "market-snapshot.v1",
        "content-factor-signal.v5.1",
        "factor.v1",
        "research-artifact.v2",
        "paper-account.v1",
        "model-artifact.v1",
    } <= entries.keys()
    for entry in entries.values():
        assert entry["formal"] is True
        assert entry["producer"]
        assert entry["consumers"]
        assert entry["checksum"].startswith("sha256:")
        assert entry["owner"]


def test_manifest_detects_checksum_duplicate_expired_and_unsafe_path(tmp_path):
    contracts = tmp_path / "contracts"
    contracts.mkdir()
    schema = contracts / "fixture.json"
    schema.write_text('{"type":"object"}\n', encoding="utf-8")
    base = {
        "name": "fixture.v1",
        "schema": "contracts/fixture.json",
        "producer": "producer",
        "consumers": ["consumer"],
        "checksum": "sha256:" + "0" * 64,
        "compatibility": "backward",
        "deprecated_at": None,
        "sunset_at": None,
        "owner": "owner",
        "formal": True,
    }
    manifest = tmp_path / "contracts" / "platform-manifest.yaml"
    manifest.write_text(yaml.safe_dump({"contracts": [base]}), encoding="utf-8")
    with pytest.raises(ManifestVerificationError, match="checksum mismatch"):
        verify_manifest(manifest)

    base["checksum"] = "sha256:" + hashlib.sha256(schema.read_bytes()).hexdigest()
    base["sunset_at"] = "2020-01-01"
    manifest.write_text(yaml.safe_dump({"contracts": [base]}), encoding="utf-8")
    with pytest.raises(ManifestVerificationError, match="expired"):
        verify_manifest(manifest, today="2026-09-04")

    base["sunset_at"] = None
    manifest.write_text(yaml.safe_dump({"contracts": [base, dict(base)]}), encoding="utf-8")
    with pytest.raises(ManifestVerificationError, match="duplicate"):
        verify_manifest(manifest)

    base["schema"] = "../outside.json"
    manifest.write_text(yaml.safe_dump({"contracts": [base]}), encoding="utf-8")
    with pytest.raises(ManifestVerificationError, match="escapes"):
        verify_manifest(manifest)


def test_schema_diff_is_deterministic_and_classifies_contract_breaks():
    old = {
        "type": "object",
        "required": ["id"],
        "properties": {
            "id": {"type": "string"},
            "mode": {"type": "string", "enum": ["A", "B"]},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "optional": {"type": "string"},
        },
    }
    new = {
        "type": "object",
        "required": ["id", "optional"],
        "properties": {
            "id": {"type": "integer"},
            "mode": {"type": "string", "enum": ["A"]},
            "score": {"type": "number", "minimum": 0.5, "maximum": 1},
            "optional": {"type": "string"},
            "new_optional": {"type": "string"},
        },
    }
    first = semantic_diff(old, new)
    second = semantic_diff(old, new)
    assert first == second
    assert first["breaking"] is True
    assert {change["kind"] for change in first["changes"]} >= {
        "requiredness_changed",
        "type_changed",
        "enum_changed",
        "range_changed",
        "field_added",
    }
    assert next(change for change in first["changes"] if change["path"] == "new_optional")["severity"] == "non_breaking"


@pytest.mark.parametrize(
    ("filename", "producer", "consumer", "version"),
    [
        ("stock_content_to_stock_factor.json", "stock_content", "stock_factor", "content-factor-signal.v5.1"),
        ("quant_to_stock_factor.json", "quant", "stock_factor", "market-snapshot.v1"),
        ("stock_factor_to_stock_agent.json", "stock_factor", "stock_agent", "factor.v1"),
        ("quant_to_stock_factor_paper.json", "quant", "stock_factor", "paper-account.v1"),
        (
            "stock_factor_to_stock_agent_research_artifact.json",
            "stock_factor",
            "stock_agent",
            "research-artifact.v2",
        ),
        ("stock_factor_to_stock_agent_model_artifact.json", "stock_factor", "stock_agent", "model-artifact.v1"),
    ],
)
def test_local_consumer_driven_fixtures_validate_against_registered_schema(filename, producer, consumer, version):
    fixture = json.loads((FIXTURES / filename).read_text(encoding="utf-8"))
    assert fixture["producer"] == producer
    assert fixture["consumer"] == consumer
    assert fixture["response"]["contract_version"] == version
    assert fixture["request"].get("contract_version", version) == version
    manifest = verify_manifest(ROOT / "contracts/platform-manifest.yaml", today=date(2026, 9, 4))
    entry = next(item for item in manifest["contracts"] if item["name"] == version)
    schema = yaml.safe_load((ROOT / entry["schema"]).read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    errors = sorted(validator.iter_errors(fixture["response"]), key=lambda error: list(error.absolute_path))
    assert not errors, "\n".join(error.message for error in errors)


def test_content_fixture_passes_formal_provider_parser(monkeypatch):
    fixture = json.loads((FIXTURES / "stock_content_to_stock_factor.json").read_text(encoding="utf-8"))
    data = fixture["response"]["data"]
    query = FormalContentQuery(**{key: data[key] for key in FormalContentQuery.model_fields})
    expected = FormalContentRef.from_query(query, data["manifest_hash"])

    def fake_post(url, **kwargs):
        return httpx.Response(200, json=fixture["response"], request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = FormalContentSignalProviderV5("http://content")
    assert provider.load_signals(["CN.A.600000"], "2026-08-10", "2026-08-10", query=query, expected_ref=expected) == []


def test_market_fixture_passes_formal_market_parser():
    fixture = json.loads((FIXTURES / "quant_to_stock_factor.json").read_text(encoding="utf-8"))
    ref = FormalMarketDatasetRef.from_payload(fixture["response"]["data"]["market_snapshot_ref"])
    snapshot = _decode_market_response(
        fixture["response"],
        formal=True,
        expected_ref=ref,
        requested_start="2026-08-10",
        requested_end="2026-08-10",
    )
    assert snapshot.formal_eligible is True
    assert snapshot.data_snapshot_id == ref.market_snapshot_id


def test_research_artifact_fixture_passes_formal_parser_and_hash():
    fixture = json.loads((FIXTURES / "stock_factor_to_stock_agent_research_artifact.json").read_text(encoding="utf-8"))
    artifact = ResearchArtifactV2.from_payload(fixture["response"])
    assert artifact.verify() is True


@pytest.mark.parametrize(
    ("filename", "path", "value"),
    [
        ("quant_to_stock_factor_paper.json", ("as_of",), "not-a-date"),
        ("quant_to_stock_factor.json", ("data", "market_snapshot_ref", "ref_hash"), "not-a-ref-hash"),
        ("stock_factor_to_stock_agent_model_artifact.json", ("model_hash",), "sha256:not-a-hash"),
        ("stock_factor_to_stock_agent_research_artifact.json", ("artifact_id",), "short"),
    ],
)
def test_registered_schema_rejects_invalid_format_and_identity_constraints(filename, path, value):
    fixture = json.loads((FIXTURES / filename).read_text(encoding="utf-8"))
    version = fixture["response"]["contract_version"]
    report = verify_manifest(ROOT / "contracts/platform-manifest.yaml", today=date(2026, 9, 4))
    entry = next(item for item in report["contracts"] if item["name"] == version)
    schema = yaml.safe_load((ROOT / entry["schema"]).read_text(encoding="utf-8"))
    body = json.loads(json.dumps(fixture["response"]))
    target = body
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    assert list(validator.iter_errors(body)), f"invalid {path} unexpectedly passed"
