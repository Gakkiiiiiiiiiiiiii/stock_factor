"""Canonical, fail-closed loader for repository configuration artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_ROOT = Path(__file__).resolve().parents[3] / "config"
# Installed wheels do not retain the repository-relative relationship between
# this module and config/.  Images provide an explicit root so runtime config
# discovery remains deterministic and cannot silently fall back elsewhere.
CONFIG_ROOT = Path(os.getenv("STOCK_FACTOR_CONFIG_ROOT", str(_DEFAULT_CONFIG_ROOT))).expanduser().resolve()
LEGACY_ROOT = CONFIG_ROOT.parent / "configs"
CHECKSUM_PREFIX = "sha256:"
SUPPORTED_SCHEMA_VERSIONS = frozenset({"config-artifact.v1", "technical-gold-set.v1"})
ENVIRONMENTS = frozenset({"shared", "dev", "test", "staging", "prod"})


class ConfigError(ValueError):
    """Raised when a configuration is missing metadata or has been tampered with."""


@dataclass(frozen=True)
class ConfigMetadata:
    schema_version: str
    config_id: str
    checksum: str
    environment: str


@dataclass(frozen=True)
class LoadedConfig:
    source: str
    content_hash: str
    metadata: ConfigMetadata
    _payload: dict[str, Any]

    @property
    def payload(self) -> dict[str, Any]:
        """Return an isolated copy so callers cannot mutate verified data."""
        return deepcopy(self._payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "content_hash": self.content_hash,
            "metadata": self.metadata.__dict__.copy(),
            "payload": self.payload,
        }


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _resolve(path: str | Path) -> Path:
    requested = Path(path)
    try:
        resolved = requested.resolve()
    except OSError as exc:
        raise ConfigError(f"configuration path is not resolvable: {path}") from exc
    if resolved == LEGACY_ROOT or LEGACY_ROOT in resolved.parents:
        warnings.warn("configs/ is deprecated; use config/", DeprecationWarning, stacklevel=3)
        raise ConfigError("legacy configs/ path is not accepted for formal configuration")
    legacy_dir = CONFIG_ROOT / "legacy"
    if resolved == legacy_dir or legacy_dir in resolved.parents:
        raise ConfigError("legacy baseline configuration is report-only and not formally loadable")
    if LEGACY_ROOT.exists():
        raise ConfigError("duplicate configuration roots detected: config/ and configs/")
    if resolved != CONFIG_ROOT and CONFIG_ROOT not in resolved.parents:
        raise ConfigError("configuration must be loaded from the repository config/ root")
    return resolved


def load_config(path: str | Path, *, environment: str | None = None) -> LoadedConfig:
    source = _resolve(path)
    if not source.is_file():
        raise ConfigError(f"configuration file does not exist: {source}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"configuration is not valid YAML/JSON: {source}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be an object")
    required = ("schema_version", "config_id", "checksum", "environment")
    if any(not isinstance(raw.get(key), str) or not raw[key].strip() for key in required):
        raise ConfigError("configuration metadata requires schema_version/config_id/checksum/environment")
    if raw["schema_version"] not in SUPPORTED_SCHEMA_VERSIONS:
        raise ConfigError("unsupported configuration schema_version")
    if raw["environment"] not in ENVIRONMENTS:
        raise ConfigError("unsupported configuration environment")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{1,127}", raw["config_id"]) is None:
        raise ConfigError("configuration config_id is invalid")
    checksum = raw["checksum"]
    payload_for_hash = {key: value for key, value in raw.items() if key != "checksum"}
    computed = CHECKSUM_PREFIX + hashlib.sha256(_canonical(payload_for_hash)).hexdigest()
    if checksum != computed:
        raise ConfigError("configuration checksum mismatch")
    if environment is not None and raw["environment"] not in {environment, "shared"}:
        raise ConfigError("configuration environment does not match requested environment")
    metadata = ConfigMetadata(*(raw[key] for key in required))
    return LoadedConfig(str(source), computed, metadata, raw)


def load_config_inventory() -> dict[str, LoadedConfig]:
    """Verify every formal config and reject duplicate identities."""
    if not CONFIG_ROOT.is_dir():
        raise ConfigError("config/ root is missing")
    inventory: dict[str, LoadedConfig] = {}
    for path in sorted(CONFIG_ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in {".yaml", ".json"} or "legacy" in path.parts:
            continue
        loaded = load_config(path)
        if loaded.metadata.config_id in inventory:
            raise ConfigError(f"duplicate config_id: {loaded.metadata.config_id}")
        inventory[loaded.metadata.config_id] = loaded
    legacy_dir = CONFIG_ROOT / "legacy"
    legacy_ids: set[str] = set()
    for sidecar in sorted(legacy_dir.glob("*.yaml.meta.json")) if legacy_dir.is_dir() else ():
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            source = sidecar.with_name(sidecar.name.removesuffix(".meta.json"))
            actual = CHECKSUM_PREFIX + hashlib.sha256(source.read_bytes()).hexdigest()
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"invalid legacy config sidecar: {sidecar}") from exc
        if metadata.get("formal_eligible") is not False or metadata.get("source_sha256") != actual:
            raise ConfigError(f"legacy config sidecar mismatch: {sidecar.name}")
        legacy_id = metadata.get("config_id")
        if not isinstance(legacy_id, str) or legacy_id in inventory or legacy_id in legacy_ids:
            raise ConfigError(f"duplicate config_id: {legacy_id or 'unknown'}")
        legacy_ids.add(legacy_id)
    return inventory


def legacy_alias(path: str | Path) -> Path:
    """Return the equivalent new path for migration tooling only.

    The alias is intentionally not accepted by ``load_config``.
    """
    requested = Path(path)
    if LEGACY_ROOT not in requested.resolve().parents and requested.resolve() != LEGACY_ROOT:
        return requested
    warnings.warn("configs/ is deprecated; migrate to config/", DeprecationWarning, stacklevel=2)
    return CONFIG_ROOT / requested.resolve().relative_to(LEGACY_ROOT)


__all__ = [
    "CONFIG_ROOT",
    "ConfigError",
    "ConfigMetadata",
    "ENVIRONMENTS",
    "LoadedConfig",
    "SUPPORTED_SCHEMA_VERSIONS",
    "legacy_alias",
    "load_config",
    "load_config_inventory",
]
