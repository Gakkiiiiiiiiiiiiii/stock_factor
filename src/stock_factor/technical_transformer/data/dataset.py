from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import build_features
from .labels import build_labels
from .schemas import CONTINUOUS_FEATURES, FEATURE_NAMES, FEATURE_SCHEMA, LABEL_SCHEMA, STATE_FEATURES, schema_hash
from .snapshot import QuantSnapshot


@dataclass(frozen=True)
class SplitConfig:
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str
    gap_days: int = 40

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "SplitConfig":
        return cls(
            train_start=str(value["train_start"]), train_end=str(value["train_end"]),
            valid_start=str(value["valid_start"]), valid_end=str(value["valid_end"]),
            test_start=str(value["test_start"]), test_end=str(value["test_end"]),
            gap_days=int(value.get("gap_days", 40)),
        )

    def ranges(self) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
        return {
            "train": (pd.Timestamp(self.train_start), pd.Timestamp(self.train_end)),
            "valid": (pd.Timestamp(self.valid_start), pd.Timestamp(self.valid_end)),
            "test": (pd.Timestamp(self.test_start), pd.Timestamp(self.test_end)),
        }


@dataclass(frozen=True)
class DatasetConfig:
    step_len: int = 128
    stride: int = 5
    min_listing_days: int = 160
    min_quality_ratio: float = 0.80
    max_symbols: int | None = None
    fit_sample_per_symbol: int = 512
    qlib_source_path: str | None = None
    split: SplitConfig | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "DatasetConfig":
        split = value.get("split") or value.get("splits")
        return cls(
            step_len=int(value.get("step_len", 128)), stride=int(value.get("stride", 5)),
            min_listing_days=int(value.get("min_listing_days", 160)),
            min_quality_ratio=float(value.get("min_quality_ratio", 0.80)),
            max_symbols=int(value["max_symbols"]) if value.get("max_symbols") else None,
            fit_sample_per_symbol=int(value.get("fit_sample_per_symbol", 512)),
            qlib_source_path=str(value["qlib_source_path"]) if value.get("qlib_source_path") else None,
            split=SplitConfig.from_mapping(split) if split else None,
        )


class RobustFeatureProcessor:
    """Train-only robust median/MAD processor for continuous features."""

    def __init__(self, clip: float = 8.0) -> None:
        self.clip = float(clip)
        self.median: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "RobustFeatureProcessor":
        if values.ndim != 2 or values.shape[1] != len(CONTINUOUS_FEATURES):
            raise ValueError("unexpected continuous feature matrix shape")
        clean = np.asarray(values, dtype=np.float64)
        clean = clean[np.isfinite(clean).all(axis=1)]
        if len(clean) == 0:
            raise ValueError("cannot fit feature processor on empty train data")
        self.median = np.nanmedian(clean, axis=0)
        q75, q25 = np.nanpercentile(clean, [75, 25], axis=0)
        self.scale = (q75 - q25) / 1.349
        self.scale = np.where(np.isfinite(self.scale) & (self.scale > 1e-6), self.scale, 1.0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.median is None or self.scale is None:
            raise RuntimeError("processor has not been fit")
        output = (np.asarray(values, dtype=np.float64) - self.median) / self.scale
        output[~np.isfinite(output)] = 0.0
        return np.clip(output, -self.clip, self.clip).astype(np.float32)

    def as_dict(self) -> dict[str, Any]:
        if self.median is None or self.scale is None:
            raise RuntimeError("processor has not been fit")
        return {"fit_scope": "train", "method": "median_mad", "clip": self.clip,
                "features": CONTINUOUS_FEATURES, "median": self.median.tolist(), "scale": self.scale.tolist()}


def _split_for_date(as_of: pd.Timestamp, split: SplitConfig | None) -> str | None:
    if split is None:
        return "train"
    for name, (start, end) in split.ranges().items():
        if start <= as_of <= end:
            return name
    return None


def _eligible_samples(
    dates: pd.Series,
    listing_days: pd.Series,
    quality: np.ndarray,
    config: DatasetConfig,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    split_ranges = config.split.ranges() if config.split else {"train": (dates.iloc[0], dates.iloc[-1])}
    last_position: dict[str, int] = {}
    for split_name, (start_date, end_date) in split_ranges.items():
        positions = [i for i, item in enumerate(dates) if start_date <= item <= end_date]
        if not positions:
            continue
        first = positions[0]
        for position in positions:
            if position - first < config.step_len - 1:
                continue
            window_start = position - config.step_len + 1
            if dates.iloc[window_start] < start_date:
                continue
            if float(listing_days.iloc[position]) < config.min_listing_days:
                continue
            quality_ratio = float(np.mean(quality[window_start:position + 1]))
            if quality_ratio < config.min_quality_ratio:
                continue
            previous = last_position.get(split_name)
            if previous is not None and position - previous < config.stride:
                continue
            samples.append({
                "end_index": int(position), "start_index": int(window_start),
                "as_of": dates.iloc[position].date().isoformat(), "split": split_name,
                "quality_ratio": quality_ratio,
            })
            last_position[split_name] = position
    return samples


def _write_qlib_compatible_layout(path: Path, series_meta: list[dict[str, Any]], all_dates: set[str]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "calendars").mkdir(exist_ok=True)
    (path / "instruments").mkdir(exist_ok=True)
    (path / "calendars" / "day.txt").write_text("\n".join(sorted(all_dates)) + "\n", encoding="utf-8")
    lines = [f"{item['symbol']}\t{item['min_date']}\t{item['max_date']}" for item in series_meta]
    (path / "instruments" / "all.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (path / "README.txt").write_text(
        "Technical Transformer Qlib-compatible calendar/instrument layout. "
        "Windows are served by TSDatasetHAdapter from immutable NPZ series.\n", encoding="utf-8"
    )


def _dump_native_qlib_provider(
    provider_path: Path,
    qlib_source_path: str | None,
    groups: list[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
) -> str | None:
    """Materialize Qlib's binary provider when quant's pinned source is available."""
    if not qlib_source_path:
        return None
    dump_script = Path(qlib_source_path) / "scripts" / "dump_bin.py"
    if not dump_script.exists():
        return None
    source_dir = provider_path.parent / "qlib_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    for symbol, group, _features, _labels in groups:
        prepared = group.copy()
        prepared["date"] = pd.to_datetime(prepared["trading_date"])
        prepared["symbol"] = symbol
        prepared["factor"] = 1.0
        prepared["change"] = prepared.groupby("symbol", sort=False)["close"].pct_change(fill_method=None).fillna(0.0)
        columns = ["symbol", "date", "open", "high", "low", "close", "volume", "amount", "turnover", "factor", "change"]
        prepared[columns].to_parquet(source_dir / f"{symbol}.parquet", index=False)
    spec = importlib.util.spec_from_file_location("technical_transformer_qlib_dump", dump_script)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dumper = module.DumpDataAll(
        data_path=str(source_dir), qlib_dir=str(provider_path), freq="day", max_workers=1,
        date_field_name="date", file_suffix=".parquet", symbol_field_name="symbol", exclude_fields="symbol",
    )
    all_datetime: set[Any] = set()
    date_range_list: list[str] = []
    for file_path in dumper.df_files:
        (begin_time, end_time), calendar_set = dumper._get_date(file_path, as_set=True, is_begin_end=True)
        all_datetime |= calendar_set
        if pd.notna(begin_time) and pd.notna(end_time):
            date_range_list.append(dumper.INSTRUMENTS_SEP.join([
                dumper.get_symbol_from_file(file_path).upper(),
                dumper._format_datetime(begin_time), dumper._format_datetime(end_time),
            ]))
    dumper._calendars_list = sorted(map(pd.Timestamp, all_datetime))
    dumper.save_calendars(dumper._calendars_list)
    dumper.save_instruments(date_range_list)
    for file_path in dumper.df_files:
        dumper._dump_bin(file_path, dumper._calendars_list)
    return "pyqlib-0.9.7"


def build_dataset(snapshot: QuantSnapshot, output_dir: str | Path, config: DatasetConfig) -> dict[str, Any]:
    if config.step_len != 128:
        raise ValueError("Technical Transformer V1 requires step_len=128")
    if config.stride != 5:
        raise ValueError("Technical Transformer V1 requires stride=5")
    output = Path(output_dir)
    series_dir = output / "series"
    if output.exists() and (output / "dataset_manifest.json").exists():
        existing = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
        if existing.get("source_market_snapshot_id") == snapshot.snapshot_id:
            return existing
        raise RuntimeError(f"dataset directory already sealed for another snapshot: {output}")
    output.mkdir(parents=True, exist_ok=True)
    series_dir.mkdir(exist_ok=True)
    frame = snapshot.frame.copy()
    symbols = sorted(frame["symbol"].astype(str).unique().tolist())
    if config.max_symbols:
        symbols = symbols[:config.max_symbols]
    frame = frame[frame["symbol"].isin(symbols)].sort_values(["symbol", "trading_date"])
    split = config.split
    # First pass: fit only on rows inside the training range.  A deterministic
    # bounded sample prevents loading millions of rows into RAM.
    fit_parts: list[np.ndarray] = []
    all_dates: set[str] = set()
    cached: list[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    for symbol, group in frame.groupby("symbol", sort=True):
        group = group.sort_values("trading_date").reset_index(drop=True)
        features = build_features(group)
        labels = build_labels(group)
        feature_valid = features[CONTINUOUS_FEATURES].notna().all(axis=1).to_numpy()
        quality = features["quality_mask"].to_numpy(dtype=float) * feature_valid.astype(float)
        features["quality_mask"] = quality
        listing = group["listing_days"] if "listing_days" in group else pd.Series(np.arange(1, len(group) + 1))
        samples = _eligible_samples(group["trading_date"], listing, quality, config)
        if split:
            train_positions = [i for i, item in enumerate(group["trading_date"]) if _split_for_date(item, split) == "train"]
        else:
            train_positions = list(range(len(group)))
        if train_positions:
            positions = train_positions[::max(1, len(train_positions) // max(config.fit_sample_per_symbol, 1))][:config.fit_sample_per_symbol]
            fit_parts.append(features.iloc[positions][CONTINUOUS_FEATURES].to_numpy(dtype=np.float32))
        cached.append((str(symbol), group, features, labels))
        all_dates.update(group["trading_date"].dt.date.astype(str).tolist())
    if not fit_parts:
        raise ValueError("no train rows available after snapshot/split filtering")
    processor = RobustFeatureProcessor().fit(np.concatenate(fit_parts, axis=0))
    samples_path = output / "samples.jsonl"
    sample_count: dict[str, int] = {"train": 0, "valid": 0, "test": 0}
    series_meta: list[dict[str, Any]] = []
    with samples_path.open("w", encoding="utf-8") as sample_file:
        for symbol, group, features, labels in cached:
            transformed = features.copy()
            transformed[CONTINUOUS_FEATURES] = processor.transform(features[CONTINUOUS_FEATURES].to_numpy(dtype=np.float32))
            transformed[STATE_FEATURES] = features[STATE_FEATURES].fillna(0.0).to_numpy(dtype=np.float32)
            x = transformed[FEATURE_NAMES].to_numpy(dtype=np.float32)
            y = labels[LABEL_SCHEMA.names].to_numpy(dtype=np.float32)
            quality = transformed["quality_mask"].to_numpy(dtype=np.float32)
            np.savez_compressed(series_dir / f"{symbol}.npz", dates=group["trading_date"].astype("int64").to_numpy(), features=x, labels=y, quality=quality)
            samples = _eligible_samples(group["trading_date"], group["listing_days"], quality, config)
            for item in samples:
                item = dict(item)
                item["symbol"] = symbol
                sample_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                sample_count[item["split"]] = sample_count.get(item["split"], 0) + 1
            series_meta.append({"symbol": symbol, "min_date": str(group["trading_date"].min().date()), "max_date": str(group["trading_date"].max().date()), "rows": len(group)})
    qlib_provider = output / "qlib_provider"
    _write_qlib_compatible_layout(qlib_provider, series_meta, all_dates)
    native_qlib_version = _dump_native_qlib_provider(qlib_provider, config.qlib_source_path, cached)
    dataset_manifest = {
        "schema_version": "technical-dataset.v1", "dataset_id": output.name,
        "source_market_snapshot_id": snapshot.snapshot_id, "source_data_version": snapshot.manifest.get("data_version"),
        "adjustment": snapshot.manifest.get("adjustment"), "frequency": snapshot.manifest.get("frequency"),
        "symbols_hash": hashlib.sha256("|".join(symbols).encode()).hexdigest(),
        "schema_hash": schema_hash({"features": FEATURE_SCHEMA, "labels": LABEL_SCHEMA.as_dict()}),
        "feature_schema_hash": schema_hash(FEATURE_SCHEMA), "label_schema_hash": schema_hash(LABEL_SCHEMA.as_dict()),
        "feature_schema_version": FEATURE_SCHEMA["schema_version"], "label_schema_version": LABEL_SCHEMA.version,
        "feature_dimensions": len(FEATURE_NAMES), "label_dimensions": len(LABEL_SCHEMA.names),
        "step_len": config.step_len, "stride": config.stride, "min_listing_days": config.min_listing_days,
        "row_count": int(len(frame)), "symbol_count": len(symbols),
        "min_date": str(frame["trading_date"].min().date()), "max_date": str(frame["trading_date"].max().date()),
        "qlib_provider_path": str(qlib_provider.resolve()), "qlib_version": native_qlib_version,
        "samples_path": str(samples_path.resolve()), "series_path": str(series_dir.resolve()),
        "sample_counts": sample_count, "processor": processor.as_dict(),
        "split": config.split.__dict__ if config.split else None,
        "created_at": pd.Timestamp.utcnow().isoformat(),
    }
    (output / "feature_schema.json").write_text(json.dumps(FEATURE_SCHEMA, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "label_schema.json").write_text(json.dumps(LABEL_SCHEMA.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "processor.json").write_text(json.dumps(processor.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return dataset_manifest


class TSDatasetHAdapter:
    """Small, on-demand TSDatasetH-shaped adapter used by the trainer.

    It keeps the Qlib contract (instrument/as-of/time-series window and
    ``step_len``) while avoiding an eager materialization of all windows.
    When Qlib is installed, this layout can be imported by a native Qlib
    DataHandler without changing the snapshot or label semantics.
    """

    def __init__(self, dataset_dir: str | Path, split: str) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.step_len = 128
        self.split = split
        self.records = [json.loads(line) for line in (self.dataset_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line]
        self.records = [item for item in self.records if item["split"] == split]
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _series(self, symbol: str) -> dict[str, np.ndarray]:
        if symbol not in self._cache:
            loaded = np.load(self.dataset_dir / "series" / f"{symbol}.npz")
            self._cache[symbol] = {key: loaded[key] for key in loaded.files}
        return self._cache[symbol]

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        item = self.records[index]
        series = self._series(item["symbol"])
        start, end = int(item["start_index"]), int(item["end_index"] + 1)
        return series["features"][start:end], series["labels"][item["end_index"]], item


class TechnicalWindowDataset(TSDatasetHAdapter):
    """PyTorch Dataset-compatible alias for the V1 trainer."""

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        return super().__getitem__(index)
