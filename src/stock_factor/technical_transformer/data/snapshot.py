from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


class SnapshotError(RuntimeError):
    """Raised when a snapshot is missing, corrupt or semantically unsafe."""


BAR_COLUMNS = ["open", "high", "low", "close", "volume", "amount", "turnover"]
PIT_COLUMNS = [
    "is_suspended", "is_st", "is_star_st", "is_delisting", "listing_days",
    "is_limit_up", "is_limit_down", "st_observed", "suspension_observed",
    "limit_observed", "delisting_observed", "turnover_observed",
    "price_observed", "volume_observed",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class QuantSnapshot:
    snapshot_id: str
    frame: pd.DataFrame
    manifest: dict[str, Any]
    manifest_path: Path
    bars_path: Path

    @classmethod
    def load(cls, snapshot_root: str | Path, snapshot_id: str, *, require_qfq: bool = True) -> "QuantSnapshot":
        root = Path(snapshot_root)
        safe_id = "".join(ch for ch in snapshot_id if ch.isalnum() or ch in "-_")
        if safe_id != snapshot_id or not snapshot_id:
            raise SnapshotError(f"invalid snapshot_id: {snapshot_id!r}")
        directory = root / snapshot_id
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise SnapshotError(f"snapshot manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files") or []
        if not files:
            raise SnapshotError(f"snapshot {snapshot_id} has no files")
        bars_path = directory / str(files[0]["path"])
        if not bars_path.exists():
            raise SnapshotError(f"snapshot bars not found: {bars_path}")
        expected = files[0].get("sha256")
        actual = sha256_file(bars_path)
        if expected and expected != actual:
            raise SnapshotError(f"snapshot sha256 mismatch for {snapshot_id}")
        if require_qfq and str(manifest.get("adjustment", "")).lower() not in {"qfq", "front"}:
            raise SnapshotError("Technical Transformer V1 requires qfq/front-adjusted prices")
        if str(manifest.get("frequency", "1d")) != "1d":
            raise SnapshotError("Technical Transformer V1 requires frequency=1d")
        frame = pd.read_parquet(bars_path)
        missing = [column for column in BAR_COLUMNS if column not in frame.columns]
        if missing:
            raise SnapshotError(f"snapshot missing required columns: {missing}")
        frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.normalize()
        frame = frame.sort_values(["symbol", "trading_date"]).reset_index(drop=True)
        if frame.duplicated(["symbol", "trading_date"]).any():
            raise SnapshotError("snapshot contains duplicate symbol/trading_date rows")
        suspended = (frame["volume"].astype(float) <= 0) | frame["close"].isna()
        for column in ["open", "high", "low", "close"]:
            if frame.loc[~suspended, column].isna().any():
                raise SnapshotError(f"snapshot contains null active-session {column}")
        for column in ["volume", "amount", "turnover"]:
            if frame[column].isna().any():
                raise SnapshotError(f"snapshot contains null {column}; do not train on incomplete market facts")
        return cls(snapshot_id, frame, manifest, manifest_path, bars_path)

    def verify(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "verified": True,
            "manifest_sha256": sha256_file(self.manifest_path),
            "bars_sha256": sha256_file(self.bars_path),
            "row_count": int(len(self.frame)),
            "min_date": str(self.frame["trading_date"].min().date()),
            "max_date": str(self.frame["trading_date"].max().date()),
        }


def write_quant_snapshot(
    frame: pd.DataFrame,
    snapshot_root: str | Path,
    *,
    source: str,
    source_data_version: str,
    adjustment: str = "qfq",
    frequency: str = "1d",
    snapshot_id: str | None = None,
    extra_manifest: dict[str, Any] | None = None,
) -> QuantSnapshot:
    """Write a content-addressed quant-compatible immutable market snapshot."""
    ordered = frame.copy()
    ordered["trading_date"] = pd.to_datetime(ordered["trading_date"]).dt.strftime("%Y-%m-%d")
    ordered = ordered.sort_values(["symbol", "trading_date"]).reset_index(drop=True)
    content = json.dumps(
        ordered.astype(object).where(ordered.notna(), None).values.tolist(),
        ensure_ascii=False, sort_keys=False, separators=(",", ":"), default=str,
    ).encode("utf-8")
    content_hash = hashlib.sha256(content).hexdigest()
    final_id = snapshot_id or f"mds-{content_hash}"
    root = Path(snapshot_root)
    directory = root / final_id
    directory.mkdir(parents=True, exist_ok=True)
    bars_path = directory / "bars.parquet"
    manifest_path = directory / "manifest.json"
    if bars_path.exists() or manifest_path.exists():
        existing = QuantSnapshot.load(root, final_id, require_qfq=False)
        if len(existing.frame) != len(ordered):
            raise SnapshotError(f"immutable conflict for {final_id}")
        return existing
    out = ordered.copy()
    out["trading_date"] = pd.to_datetime(out["trading_date"])
    out.to_parquet(bars_path, index=False)
    manifest: dict[str, Any] = {
        "schema_version": "market-snapshot.v1",
        "snapshot_id": final_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_version": source_data_version,
        "source": source,
        "source_provider": source,
        "source_data_version": source_data_version,
        "frequency": frequency,
        "adjustment": adjustment,
        "start": str(out["trading_date"].min().date()),
        "end": str(out["trading_date"].max().date()),
        "as_of": str(out["trading_date"].max().date()),
        "symbols": sorted(out["symbol"].astype(str).unique().tolist()),
        "fields": list(out.columns),
        "row_count": int(len(out)),
        "min_date": str(out["trading_date"].min().date()),
        "max_date": str(out["trading_date"].max().date()),
        "pit_enforced": True,
        "files": [{"path": "bars.parquet", "sha256": sha256_file(bars_path)}],
    }
    manifest.update(extra_manifest or {})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return QuantSnapshot.load(root, final_id, require_qfq=False)
