from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from stock_factor.cli import main
from stock_factor.config.schema import ConfigError, legacy_alias, load_config, load_config_inventory


def _write_config(path: Path, *, environment: str = "shared") -> None:
    payload = {
        "schema_version": "config-artifact.v1",
        "config_id": "fixture",
        "environment": environment,
        "value": {"threshold": 1},
    }
    payload["checksum"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_repository_has_one_config_root_and_all_artifacts_verify():
    assert not Path("configs").exists()
    for path in Path("config").rglob("*"):
        if path.is_file() and path.suffix in {".yaml", ".json"} and "legacy" not in path.parts:
            assert load_config(path).content_hash.startswith("sha256:")
    for legacy in Path("config/legacy").glob("*.yaml"):
        with pytest.raises(ConfigError, match="legacy"):
            load_config(legacy)


def test_config_loader_reports_source_hash_and_rejects_tamper(tmp_path):
    path = tmp_path / "fixture.yaml"
    _write_config(path)
    # The loader intentionally accepts only the repository root.
    import stock_factor.config.schema as schema

    monkey_root = schema.CONFIG_ROOT
    monkeypatch_root = tmp_path / "config"
    monkeypatch_root.mkdir()
    target = monkeypatch_root / path.name
    target.write_bytes(path.read_bytes())
    schema.CONFIG_ROOT = monkeypatch_root
    schema.LEGACY_ROOT = tmp_path / "configs"
    try:
        loaded = load_config(target)
        assert loaded.source.endswith("fixture.yaml")
        assert loaded.content_hash == loaded.metadata.checksum
        loaded.payload["value"]["threshold"] = 99
        assert loaded.payload["value"]["threshold"] == 1
        target.write_text(target.read_text(encoding="utf-8").replace("threshold: 1", "threshold: 2"), encoding="utf-8")
        with pytest.raises(ConfigError, match="checksum mismatch"):
            load_config(target)
    finally:
        schema.CONFIG_ROOT = monkey_root
        schema.LEGACY_ROOT = monkey_root.parent / "configs"


def test_legacy_alias_warns_but_formal_loader_rejects(tmp_path):
    import stock_factor.config.schema as schema

    original_root, original_legacy = schema.CONFIG_ROOT, schema.LEGACY_ROOT
    schema.CONFIG_ROOT = tmp_path / "config"
    schema.LEGACY_ROOT = tmp_path / "configs"
    schema.CONFIG_ROOT.mkdir()
    schema.LEGACY_ROOT.mkdir()
    legacy = schema.LEGACY_ROOT / "old.yaml"
    _write_config(schema.CONFIG_ROOT / "old.yaml")
    legacy.write_bytes((schema.CONFIG_ROOT / "old.yaml").read_bytes())
    try:
        with pytest.raises(ConfigError, match="duplicate"):
            load_config(schema.CONFIG_ROOT / "old.yaml")
        with pytest.warns(DeprecationWarning):
            assert legacy_alias(legacy) == schema.CONFIG_ROOT / "old.yaml"
        with pytest.warns(DeprecationWarning), pytest.raises(ConfigError, match="legacy"):
            load_config(legacy)
    finally:
        schema.CONFIG_ROOT, schema.LEGACY_ROOT = original_root, original_legacy


def test_cli_config_inspect_and_verify(capsys):
    path = Path("config/technical_v1.yaml")
    assert main(["research", "config-inspect", "--path", str(path)]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["source"].endswith("technical_v1.yaml")
    assert main(["research", "config-verify", "--path", str(path)]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["verified"] is True
    assert main(["experimental"]) == 0
    assert json.loads(capsys.readouterr().out)["formal_eligible"] is False


def test_config_schema_environment_and_identity_validation(tmp_path):
    import stock_factor.config.schema as schema

    original_root, original_legacy = schema.CONFIG_ROOT, schema.LEGACY_ROOT
    root = tmp_path / "config"
    root.mkdir()
    schema.CONFIG_ROOT, schema.LEGACY_ROOT = root, tmp_path / "configs"
    try:
        unknown = root / "unknown.yaml"
        _write_config(unknown)
        unknown.write_text(
            unknown.read_text(encoding="utf-8").replace("config-artifact.v1", "config-unknown.v1"), encoding="utf-8"
        )
        with pytest.raises(ConfigError, match="schema_version|checksum"):
            load_config(unknown)
        bad_env = root / "bad.yaml"
        _write_config(bad_env, environment="unknown")
        with pytest.raises(ConfigError, match="environment"):
            load_config(bad_env)
        bad_env.unlink()
        first = root / "first.yaml"
        second = root / "second.yaml"
        _write_config(first)
        _write_config(second)
        with pytest.raises(ConfigError, match="duplicate config_id"):
            load_config_inventory()
    finally:
        schema.CONFIG_ROOT, schema.LEGACY_ROOT = original_root, original_legacy
