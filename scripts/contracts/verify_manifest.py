"""Verify the repository's formal contract inventory.

The verifier intentionally has no network or cross-repository behavior.  The
manifest is the reviewable source of inventory metadata while each checksum is
calculated from the checked-in schema bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMPATIBILITY = {"backward", "forward", "full", "none"}


class ManifestVerificationError(ValueError):
    """Raised when a formal contract inventory is not safe to consume."""


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ManifestVerificationError(f"manifest cannot be read: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestVerificationError("manifest root must be an object")
    return payload


def _date(value: Any, field: str, contract: str) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ManifestVerificationError(f"{contract}: {field} must be an ISO date or null")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ManifestVerificationError(f"{contract}: {field} must be an ISO date") from exc


def _schema_path(raw: Any, *, root: Path, contract: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ManifestVerificationError(f"{contract}: schema path is required")
    candidate = Path(raw)
    if candidate.is_absolute():
        raise ManifestVerificationError(f"{contract}: schema path must be repository-relative")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ManifestVerificationError(f"{contract}: schema path escapes repository: {raw}") from exc
    if not resolved.is_file():
        raise ManifestVerificationError(f"{contract}: schema file is missing: {raw}")
    return resolved


def verify_manifest(manifest_path: str | Path, *, today: date | str | None = None) -> dict[str, Any]:
    """Return a deterministic verification report or raise on any violation."""

    path = Path(manifest_path).resolve()
    root = path.parent.parent
    payload = _load(path)
    entries = payload.get("contracts")
    if not isinstance(entries, list) or not entries:
        raise ManifestVerificationError("manifest contracts must be a non-empty list")

    if today is None:
        check_date = date.today()
    elif isinstance(today, date):
        check_date = today
    else:
        check_date = _date(today, "today", "manifest")
        assert check_date is not None

    seen: set[str] = set()
    verified: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ManifestVerificationError("each manifest contract must be an object")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ManifestVerificationError("contract name is required")
        if name in seen:
            raise ManifestVerificationError(f"duplicate contract name: {name}")
        seen.add(name)

        producer = entry.get("producer")
        consumers = entry.get("consumers")
        owner = entry.get("owner")
        if not isinstance(producer, str) or not producer.strip():
            raise ManifestVerificationError(f"{name}: producer is required")
        if (
            not isinstance(consumers, list)
            or not consumers
            or not all(isinstance(c, str) and c.strip() for c in consumers)
        ):
            raise ManifestVerificationError(f"{name}: consumers must be a non-empty list")
        if len(set(consumers)) != len(consumers):
            raise ManifestVerificationError(f"{name}: duplicate consumer")
        if not isinstance(owner, str) or not owner.strip():
            raise ManifestVerificationError(f"{name}: owner is required")
        if entry.get("formal") is not True:
            raise ManifestVerificationError(f"{name}: formal contracts must set formal: true")
        compatibility = entry.get("compatibility")
        if compatibility not in _COMPATIBILITY:
            raise ManifestVerificationError(f"{name}: unsupported compatibility: {compatibility!r}")
        deprecated_at = _date(entry.get("deprecated_at"), "deprecated_at", name)
        sunset_at = _date(entry.get("sunset_at"), "sunset_at", name)
        if deprecated_at and sunset_at and sunset_at < deprecated_at:
            raise ManifestVerificationError(f"{name}: sunset_at precedes deprecated_at")
        if sunset_at and sunset_at <= check_date:
            raise ManifestVerificationError(f"{name}: formal contract is expired at {sunset_at.isoformat()}")

        schema = _schema_path(entry.get("schema"), root=root, contract=name)
        actual = "sha256:" + hashlib.sha256(schema.read_bytes()).hexdigest()
        expected = entry.get("checksum")
        if not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
            raise ManifestVerificationError(f"{name}: checksum must be sha256:<64 lowercase hex>")
        if actual != expected:
            raise ManifestVerificationError(f"{name}: checksum mismatch (expected {expected}, actual {actual})")
        verified.append(
            {
                "name": name,
                "schema": str(schema.relative_to(root)).replace("\\", "/"),
                "producer": producer,
                "consumers": sorted(consumers),
                "checksum": actual,
                "compatibility": compatibility,
                "deprecated_at": deprecated_at.isoformat() if deprecated_at else None,
                "sunset_at": sunset_at.isoformat() if sunset_at else None,
                "owner": owner,
                "formal": True,
            }
        )

    verified.sort(key=lambda item: item["name"])
    return {"manifest": str(path), "checked_at": check_date.isoformat(), "contracts": verified, "valid": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="contracts/platform-manifest.yaml")
    parser.add_argument("--today", help="override validation date (YYYY-MM-DD)")
    args = parser.parse_args(argv)
    try:
        report = verify_manifest(args.manifest, today=args.today)
    except ManifestVerificationError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI workflow
    sys.exit(main())
