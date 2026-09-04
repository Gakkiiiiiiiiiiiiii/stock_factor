"""Non-mutating readiness and admission checks for research workflows."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Callable, Mapping

import yaml
from sqlalchemy import inspect, text

from stock_factor.config.runtime import RuntimeConfig
from stock_factor.config.schema import CONFIG_ROOT

READINESS_POLICY_VERSION = "readiness_v1"
MAX_SNAPSHOT_AGE_SECONDS = 86_400


class ReadinessAdmissionError(RuntimeError):
    code = "RESEARCH_NOT_READY"

    def __init__(self, report: "ReadinessReport") -> None:
        self.report = report
        super().__init__(f"{self.code}: {', '.join(report.blocking_reasons)}")


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _contract_inventory() -> dict[str, Any]:
    root = CONFIG_ROOT.parent / "contracts"
    items: dict[str, str] = {}
    if root.exists():
        for path in sorted(root.rglob("*")):
            if path.is_file():
                items[str(path.relative_to(root)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = root / "platform-manifest.yaml"
    errors: list[str] = []
    registered: dict[str, str] = {}
    manifest_version = None
    if not manifest_path.is_file():
        errors.append("manifest_missing")
    else:
        try:
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
            manifest_version = manifest.get("manifest_version")
            for entry in manifest.get("contracts", []):
                name, schema, expected = entry.get("name"), entry.get("schema"), entry.get("checksum")
                path = root.parent / str(schema) if schema else None
                actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None
                if not name or not schema or not expected or actual != expected:
                    errors.append(f"checksum_mismatch:{name or 'unknown'}")
                else:
                    registered[str(name)] = actual
                if "sunset_at" not in entry:
                    errors.append(f"sunset_missing:{name or 'unknown'}")
        except Exception as exc:  # noqa: BLE001
            errors.append(type(exc).__name__)
    return {
        "revision": os.getenv("FACTOR_CONTRACT_REVISION", "working-tree"),
        "manifest_version": manifest_version,
        "checksums": items,
        "registered_checksums": registered,
        "valid": not errors,
        "errors": errors,
    }


def _immutable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _immutable(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_immutable(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class ReadinessReport:
    ready: bool
    runtime_profile: str
    checks: Mapping[str, Any]
    blocking_reasons: tuple[str, ...]
    threshold_version: str = READINESS_POLICY_VERSION
    frozen_at: str = ""
    evidence_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", _immutable(self.checks))
        frozen_at = self.frozen_at or _utc_now()
        object.__setattr__(self, "frozen_at", frozen_at)
        payload = self.to_dict(include_hash=False)
        object.__setattr__(self, "evidence_hash", self.evidence_hash or _sha256_payload(payload))

    def _identity_payload(self) -> dict[str, Any]:
        """Return the frozen check identity, excluding observation time.

        ``frozen_at`` is retained in the evidence for auditability, while the
        hash is stable across an immediate, unchanged readiness revalidation.
        """
        return {
            "ready": self.ready,
            "runtime_profile": self.runtime_profile,
            "checks": _plain(self.checks),
            "blocking_reasons": list(self.blocking_reasons),
            "threshold_version": self.threshold_version,
        }

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "ready": self.ready,
            "runtime_profile": self.runtime_profile,
            "checks": _plain(self.checks),
            "blocking_reasons": list(self.blocking_reasons),
            "threshold_version": self.threshold_version,
            "frozen_at": self.frozen_at,
        }
        if include_hash:
            payload["evidence_hash"] = self.evidence_hash
        return payload

    def verify(self) -> bool:
        return self.evidence_hash == _sha256_payload(self.to_dict(include_hash=False))

    def _sha256_identity(self) -> str:
        return _sha256_payload(self._identity_payload())


class ReadinessService:
    """Evaluate readiness without probing endpoints that can mutate state."""

    def __init__(
        self,
        runtime_config: RuntimeConfig,
        *,
        database: Any | None = None,
        artifact_store: Any | None = None,
        oos_repository: Any | None = None,
        model_registry: Any | None = None,
        market_probe: Callable[[], Mapping[str, Any]] | None = None,
        content_probe: Callable[[], Mapping[str, Any]] | None = None,
        resource_probe: Callable[[], Mapping[str, Any]] | None = None,
        upstream_probe: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self.config = runtime_config
        self.database = database
        self.artifact_store = artifact_store
        self.oos_repository = oos_repository
        self.model_registry = model_registry
        self.market_probe = market_probe
        self.content_probe = content_probe
        self.resource_probe = resource_probe
        self.upstream_probe = upstream_probe

    def liveness(self) -> dict[str, Any]:
        return {"ready": True, "status": "alive", "runtime_profile": self.config.profile.value}

    def research(self, *, model_requested: bool = False) -> ReadinessReport:
        checks: dict[str, Any] = {
            "runtime": {"profile": self.config.profile.value},
            "market_snapshot_authority": self._market_check(),
            "content_v5_authority": self._content_check(),
            "contracts": _contract_inventory(),
            "database": self._database_check(),
            "artifact_store": self._artifact_check(),
            "oos_scheduler": self._oos_check(),
            "resources": self._resource_check(),
        }
        if model_requested:
            checks["model_registry"] = self._model_check()
        reasons: list[str] = []
        if checks["market_snapshot_authority"]["status"] != "READY":
            reasons.append("MARKET_SNAPSHOT_NOT_READY")
        if checks["content_v5_authority"]["status"] != "READY":
            reasons.append("CONTENT_V5_NOT_READY")
        if not checks["contracts"]["valid"]:
            reasons.append("CONTRACT_INVENTORY_MISSING")
        if checks["database"]["status"] != "READY":
            reasons.append("DATABASE_SCHEMA_NOT_READY")
        if checks["artifact_store"]["status"] != "READY":
            reasons.append("ARTIFACT_STORE_NOT_READY")
        if checks["oos_scheduler"]["status"] != "READY":
            reasons.append("OOS_LEASE_NOT_READY")
        if checks["resources"]["status"] != "READY":
            reasons.append("RESOURCE_ADMISSION_NOT_READY")
        if model_requested and checks["model_registry"]["status"] != "READY":
            reasons.append("MODEL_REGISTRY_NOT_READY")
        return ReadinessReport(not reasons, self.config.profile.value, checks, tuple(reasons))

    def ml(self) -> ReadinessReport:
        checks = {
            "torch": {"installed": importlib.util.find_spec("torch") is not None},
            "runtime_profile": self.config.profile.value,
            "model_registry": self._model_check(),
        }
        reasons: list[str] = []
        if not checks["torch"]["installed"]:
            reasons.append("TORCH_NOT_INSTALLED")
        if checks["model_registry"]["status"] != "READY":
            reasons.append("SEALED_RELIABLE_MODEL_NOT_READY")
        return ReadinessReport(not reasons, self.config.profile.value, checks, tuple(reasons))

    def paper(self) -> ReadinessReport:
        try:
            upstream = dict(self.upstream_probe() if self.upstream_probe else {"status": "NOT_PROBED"})
        except Exception as exc:  # noqa: BLE001
            upstream = {"status": "NOT_READY", "reason": type(exc).__name__}
        checks = {
            "authority": self.config.paper_authority.value,
            "contract": self.config.required_quant_contract,
            "checksum_configured": bool(self.config.required_quant_checksum),
            "upstream": upstream,
        }
        reasons: list[str] = []
        if self.config.paper_authority.value != "quant":
            reasons.append("PAPER_AUTHORITY_NOT_QUANT")
        if self.config.required_quant_contract != "paper-account.v1":
            reasons.append("PAPER_CONTRACT_MISMATCH")
        if not self.config.required_quant_checksum:
            reasons.append("PAPER_CONTRACT_CHECKSUM_MISSING")
        if checks["upstream"].get("status") != "READY":
            reasons.append("QUANT_UPSTREAM_NOT_REACHABLE")
        return ReadinessReport(not reasons, self.config.profile.value, checks, tuple(reasons))

    def admit_formal_mining(self, payload: Mapping[str, Any]) -> ReadinessReport:
        report = self.research(model_requested=bool(payload.get("use_model")))
        if not report.ready:
            raise ReadinessAdmissionError(report)
        return report

    def revalidate_oos(self, payload: Mapping[str, Any], evidence_hash: str | None) -> ReadinessReport:
        report = self.research(model_requested=bool(payload.get("use_model")))
        frozen = payload.get("readiness_evidence")
        prior_identity = None
        frozen_integrity = False
        if isinstance(frozen, Mapping):
            frozen_without_hash = {key: value for key, value in frozen.items() if key != "evidence_hash"}
            frozen_integrity = bool(frozen.get("evidence_hash")) and frozen.get("evidence_hash") == _sha256_payload(
                frozen_without_hash
            )
            prior_identity = _sha256_payload(
                {
                    "ready": frozen.get("ready"),
                    "runtime_profile": frozen.get("runtime_profile"),
                    "checks": frozen.get("checks", {}),
                    "blocking_reasons": frozen.get("blocking_reasons", []),
                    "threshold_version": frozen.get("threshold_version"),
                }
            )
        stale = (
            not isinstance(frozen, Mapping)
            or not evidence_hash
            or not frozen_integrity
            or frozen.get("evidence_hash") != evidence_hash
            or not report.ready
            or report._sha256_identity() != prior_identity
        )
        if stale:
            if report.ready:
                report = ReadinessReport(
                    False,
                    report.runtime_profile,
                    report.checks,
                    ("READINESS_EVIDENCE_STALE",),
                    report.threshold_version,
                )
            raise ReadinessAdmissionError(report)
        return report

    def _market_check(self) -> dict[str, Any]:
        try:
            result = dict(self.market_probe() if self.market_probe else {})
        except Exception as exc:  # noqa: BLE001
            return {"status": "NOT_READY", "reason": type(exc).__name__}
        result.setdefault("authority", "quant" if self.config.paper_authority.value == "quant" else "unknown")
        result.setdefault("contract", "market-snapshot.v1")
        result["freshness"] = self._freshness(result)
        result["status"] = (
            "READY" if result.get("authority") == "quant" and result.get("freshness") == "READY" else "NOT_READY"
        )
        return result

    def _content_check(self) -> dict[str, Any]:
        try:
            result = dict(self.content_probe() if self.content_probe else {})
        except Exception as exc:  # noqa: BLE001
            return {"status": "NOT_READY", "reason": type(exc).__name__}
        result.setdefault("authority", "stock_content")
        result.setdefault("contract", self.config.required_content_contract)
        result["freshness"] = self._freshness(result)
        result["status"] = (
            "READY"
            if result.get("contract") == "content-factor-signal.v5.1" and result.get("freshness") == "READY"
            else "NOT_READY"
        )
        return result

    @staticmethod
    def _freshness(result: Mapping[str, Any]) -> str:
        """Require an explicit UTC observation/as-of pair and bounded age."""
        observed = result.get("observed_at")
        as_of = result.get("as_of")
        try:
            observed_dt = datetime.fromisoformat(str(observed).replace("Z", "+00:00"))
            as_of_dt = datetime.fromisoformat(str(as_of).replace("Z", "+00:00"))
            if observed_dt.tzinfo is None or as_of_dt.tzinfo is None:
                return "STALE"
            age = (datetime.now(UTC) - observed_dt.astimezone(UTC)).total_seconds()
            if age < 0 or age > MAX_SNAPSHOT_AGE_SECONDS or as_of_dt.astimezone(UTC) > observed_dt.astimezone(UTC):
                return "STALE"
            return "READY"
        except (TypeError, ValueError):
            return "STALE"

    def _database_check(self) -> dict[str, Any]:
        if self.database is None:
            return {"status": "NOT_READY", "schema_version": "unknown", "migration_version": "unknown"}
        try:
            with self.database.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            tables = set(inspect(self.database.engine).get_table_names())
            required_tables = {
                "factor_definition",
                "factor_job",
                "oos_evaluation_runs",
                "oos_candidate_checkpoints",
                "oos_cohort_artifacts",
                "research_artifacts_v2",
            }
            missing = sorted(required_tables - tables)
            if missing:
                return {
                    "status": "NOT_READY",
                    "schema_version": "incomplete",
                    "migration_version": os.getenv("FACTOR_DB_MIGRATION_VERSION", "metadata-v1"),
                    "missing_tables": missing,
                }
            return {
                "status": "READY",
                "schema_version": "metadata-v1",
                "migration_version": os.getenv("FACTOR_DB_MIGRATION_VERSION", "metadata-v1"),
            }
        except Exception as exc:  # noqa: BLE001
            return {"status": "NOT_READY", "reason": type(exc).__name__}

    def _artifact_check(self) -> dict[str, Any]:
        append_only = (
            bool(getattr(self.artifact_store, "append_only", False)) if self.artifact_store is not None else False
        )
        return {
            "status": "READY" if self.artifact_store is not None and append_only else "NOT_READY",
            "append_only": append_only,
        }

    def _oos_check(self) -> dict[str, Any]:
        required = ("start_or_resume", "renew", "put_checkpoint", "seal")
        available = self.oos_repository is not None and all(hasattr(self.oos_repository, name) for name in required)
        return {"status": "READY" if available else "NOT_READY", "lease_capable": available}

    def _resource_check(self) -> dict[str, Any]:
        try:
            result = dict(
                self.resource_probe()
                if self.resource_probe
                else {"queue_depth": 0, "deadline_seconds": 60, "memory_mb": 1024}
            )
            queue_depth = float(result.get("queue_depth", 0))
            deadline = float(result.get("deadline_seconds", 0))
            memory = float(result.get("memory_mb", 0))
            ready = (
                all(math.isfinite(value) for value in (queue_depth, deadline, memory))
                and queue_depth <= 100
                and deadline >= 30
                and memory >= 512
            )
        except (TypeError, ValueError):
            return {"status": "NOT_READY", "reason": "INVALID_RESOURCE_PROBE"}
        except Exception as exc:  # noqa: BLE001
            return {"status": "NOT_READY", "reason": type(exc).__name__}
        result["status"] = "READY" if ready else "NOT_READY"
        return result

    def _model_check(self) -> dict[str, Any]:
        if self.model_registry is None:
            return {"status": "NOT_READY", "reason": "registry_missing"}
        try:
            records = self.model_registry.list()
            ready = any(
                getattr(getattr(item, "status", None), "value", getattr(item, "status", None)) == "PROMOTED"
                and (getattr(item, "reliability_seal", {}) or {}).get("status") == "PASS"
                and bool(getattr(item, "reliability_report_hash", ""))
                and bool(getattr(item, "dependency_lock_hash", ""))
                and bool((getattr(item, "hardware_profile", {}) or {}).get("device"))
                for item in records
            )
            return {"status": "READY" if ready else "NOT_READY", "promoted_reliable": ready}
        except Exception as exc:  # noqa: BLE001
            return {"status": "NOT_READY", "reason": type(exc).__name__}


__all__ = ["ReadinessAdmissionError", "ReadinessReport", "ReadinessService", "READINESS_POLICY_VERSION"]
