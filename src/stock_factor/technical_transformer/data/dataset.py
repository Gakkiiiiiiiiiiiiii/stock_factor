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


CANONICAL_SPLITS = ("train", "valid", "time_test", "instrument_test", "double_oos")
TURNOVER_DERIVED_FEATURES = {"turnover", "turnover_ratio_5", "turnover_ratio_20", "turnover_zscore_20"}
OPTIONAL_ROLLING_FEATURES = {"volume_zscore_20", "volume_zscore_60", *TURNOVER_DERIVED_FEATURES}


def canonical_split_name(name: str) -> str:
    return "time_test" if name == "test" else name


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
        result = {
            "train": (pd.Timestamp(self.train_start), pd.Timestamp(self.train_end)),
            "valid": (pd.Timestamp(self.valid_start), pd.Timestamp(self.valid_end)),
            "test": (pd.Timestamp(self.test_start), pd.Timestamp(self.test_end)),
        }
        ordered = list(result.values())
        for (previous_start, previous_end), (next_start, next_end) in zip(ordered, ordered[1:]):
            if next_start <= previous_end:
                raise ValueError("time split ranges overlap")
            if (next_start - previous_end).days < self.gap_days:
                raise ValueError(f"time split gap is smaller than gap_days={self.gap_days}")
            if next_end < next_start or previous_end < previous_start:
                raise ValueError("time split range has inverted dates")
        return result


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
    holdout_ratio: float = 0.20
    symbol_split_seed: int = 42
    stratify_fields: tuple[str, ...] = ("board", "industry", "market_cap_bucket", "liquidity_bucket")

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
            holdout_ratio=float(value.get("holdout_ratio", 0.20)),
            symbol_split_seed=int(value.get("symbol_split_seed", 42)),
            stratify_fields=tuple(str(item) for item in value.get("stratify_fields", ("board", "industry", "market_cap_bucket", "liquidity_bucket"))),
        )


def deterministic_symbol_split(
    symbols: list[str] | tuple[str, ...],
    metadata: pd.DataFrame | None = None,
    *,
    holdout_ratio: float = 0.20,
    seed: int = 42,
    stratify_fields: tuple[str, ...] = (),
) -> tuple[list[str], list[str]]:
    """Return a reproducible, symbol-disjoint train/holdout split.

    If optional point-in-time symbol metadata is available, symbols are
    allocated within deterministic strata.  Otherwise the same hash ordering
    is used globally; sorting symbols alone is explicitly avoided.
    """
    unique = sorted({str(symbol) for symbol in symbols})
    if not unique:
        return [], []
    ratio = min(max(float(holdout_ratio), 0.0), 0.95)
    rows: dict[str, tuple[str, ...]] = {}
    available = set(metadata.columns) if metadata is not None else set()
    if metadata is not None and "symbol" in available:
        meta = metadata.drop_duplicates("symbol").set_index("symbol")
        for symbol in unique:
            if symbol in meta.index:
                rows[symbol] = tuple(str(meta.loc[symbol, field]) if field in available and pd.notna(meta.loc[symbol, field]) else "UNKNOWN" for field in stratify_fields)
    groups: dict[tuple[str, ...], list[str]] = {}
    for symbol in unique:
        groups.setdefault(rows.get(symbol, ()), []).append(symbol)
    holdout: list[str] = []
    for group_key in sorted(groups, key=repr):
        group = groups[group_key]
        desired = int(round(len(group) * ratio))
        if ratio > 0 and len(group) >= 5:
            desired = max(1, desired)
        desired = min(desired, max(0, len(group) - 1))
        ranked = sorted(group, key=lambda item: hashlib.sha256(f"{seed}:{repr(group_key)}:{item}".encode()).hexdigest())
        holdout.extend(ranked[:desired])
    # Small datasets must still have both sides when a non-zero holdout was requested.
    if ratio > 0 and len(unique) >= 2 and not holdout:
        holdout.append(min(unique, key=lambda item: hashlib.sha256(f"{seed}:{item}".encode()).hexdigest()))
    if len(holdout) >= len(unique):
        holdout = holdout[:-1]
    holdout_set = set(holdout)
    return [symbol for symbol in unique if symbol not in holdout_set], [symbol for symbol in unique if symbol in holdout_set]


def symbol_hash(symbols: list[str] | tuple[str, ...]) -> str:
    return hashlib.sha256("|".join(sorted(map(str, symbols))).encode()).hexdigest()


def assert_split_disjoint(records: list[dict[str, Any]]) -> None:
    """Validate no symbol/as-of sample is assigned to two evaluation splits."""
    seen: dict[tuple[str, str], str] = {}
    for item in records:
        key = (str(item["symbol"]), str(item["as_of"]))
        split = canonical_split_name(str(item["split"]))
        previous = seen.get(key)
        if previous is not None and previous != split:
            raise ValueError(f"sample appears in multiple splits: {key} ({previous}, {split})")
        seen[key] = split


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
        if not np.isfinite(clean).any():
            raise ValueError("cannot fit feature processor on empty train data")
        # Fit each channel on the finite observations available in the train
        # range; a constant/unknown rolling statistic gets a neutral fallback.
        self.median = np.zeros(clean.shape[1], dtype=np.float64)
        q75 = np.zeros(clean.shape[1], dtype=np.float64)
        q25 = np.zeros(clean.shape[1], dtype=np.float64)
        for index in range(clean.shape[1]):
            values = clean[np.isfinite(clean[:, index]), index]
            if len(values):
                self.median[index] = float(np.median(values))
                q25[index], q75[index] = np.percentile(values, [25, 75])
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


def _eligible_samples_for_range(
    dates: pd.Series,
    listing_days: pd.Series,
    quality: np.ndarray,
    config: DatasetConfig,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    split_name: str,
) -> list[dict[str, Any]]:
    """Create samples for one explicit range without crossing its start."""
    positions = [i for i, item in enumerate(dates) if start_date <= item <= end_date]
    samples: list[dict[str, Any]] = []
    last_position: int | None = None
    if not positions:
        return samples
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
        if last_position is not None and position - last_position < config.stride:
            continue
        samples.append({
            "end_index": int(position), "start_index": int(window_start),
            "as_of": dates.iloc[position].date().isoformat(), "split": split_name,
            "quality_ratio": quality_ratio,
        })
        last_position = position
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
            if existing.get("feature_schema_version") != FEATURE_SCHEMA["schema_version"] or existing.get("label_schema_version") != LABEL_SCHEMA.version:
                raise RuntimeError(f"dataset directory is sealed with an older schema; choose a new V2 output directory: {output}")
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
    metadata_columns = ["symbol", *config.stratify_fields]
    metadata = frame[[column for column in metadata_columns if column in frame.columns]].drop_duplicates("symbol") if "symbol" in frame else None
    train_symbols, holdout_symbols = deterministic_symbol_split(
        symbols, metadata, holdout_ratio=config.holdout_ratio, seed=config.symbol_split_seed,
        stratify_fields=config.stratify_fields,
    )
    holdout_set = set(holdout_symbols)

    # Fit only on train-symbol/train-time rows.  Rows with missing turnover are
    # allowed into the calendar; their turnover-derived continuous values are
    # imputed by the train-only processor and flagged in the state channels.
    fit_parts: list[np.ndarray] = []
    all_dates: set[str] = set()
    cached: list[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    required_features = [name for name in CONTINUOUS_FEATURES if name not in OPTIONAL_ROLLING_FEATURES]
    train_range = split.ranges().get("train") if split else None
    for symbol, group in frame.groupby("symbol", sort=True):
        group = group.sort_values("trading_date").reset_index(drop=True)
        features = build_features(group)
        labels = build_labels(group)
        feature_valid = features[required_features].notna().all(axis=1).to_numpy()
        quality = features["quality_mask"].to_numpy(dtype=float) * feature_valid.astype(float)
        features["quality_mask"] = quality
        listing = group["listing_days"] if "listing_days" in group else pd.Series(np.arange(1, len(group) + 1))
        train_positions = list(range(len(group)))
        if train_range:
            train_positions = [i for i, item in enumerate(group["trading_date"]) if train_range[0] <= item <= train_range[1]]
        if symbol not in holdout_set and train_positions:
            sample_count = min(len(train_positions), max(config.fit_sample_per_symbol, 1))
            positions = [train_positions[index] for index in np.linspace(0, len(train_positions) - 1, sample_count, dtype=int)]
            fit_values = features.iloc[positions][CONTINUOUS_FEATURES].to_numpy(dtype=np.float32)
            required_indices = [CONTINUOUS_FEATURES.index(name) for name in required_features]
            finite = np.isfinite(fit_values[:, required_indices]).all(axis=1)
            if finite.any():
                fit_parts.append(fit_values[finite])
        cached.append((str(symbol), group, features, labels))
        all_dates.update(group["trading_date"].dt.date.astype(str).tolist())
    if not fit_parts:
        raise ValueError("no train rows available after snapshot/split filtering")
    processor = RobustFeatureProcessor().fit(np.concatenate(fit_parts, axis=0))
    samples_path = output / "samples.jsonl"
    sample_count: dict[str, int] = {name: 0 for name in CANONICAL_SPLITS}
    all_records: list[dict[str, Any]] = []
    series_meta: list[dict[str, Any]] = []
    ranges = split.ranges() if split else {}
    with samples_path.open("w", encoding="utf-8") as sample_file:
        for symbol, group, features, labels in cached:
            transformed = features.copy()
            transformed[CONTINUOUS_FEATURES] = processor.transform(features[CONTINUOUS_FEATURES].to_numpy(dtype=np.float32))
            transformed[STATE_FEATURES] = features[STATE_FEATURES].fillna(0.0).to_numpy(dtype=np.float32)
            x = transformed[FEATURE_NAMES].to_numpy(dtype=np.float32)
            y = labels[LABEL_SCHEMA.names].to_numpy(dtype=np.float32)
            quality = transformed["quality_mask"].to_numpy(dtype=np.float32)
            np.savez_compressed(
                series_dir / f"{symbol}.npz", dates=group["trading_date"].astype("int64").to_numpy(),
                features=x, labels=y, quality=quality,
            )
            listing = group["listing_days"] if "listing_days" in group else pd.Series(np.arange(1, len(group) + 1))
            symbol_records: list[dict[str, Any]] = []
            if symbol not in holdout_set:
                base_samples = _eligible_samples(group["trading_date"], listing, quality, config)
                for item in base_samples:
                    item = dict(item)
                    item["split"] = "time_test" if item["split"] == "test" else item["split"]
                    symbol_records.append(item)
            elif ranges:
                if "valid" in ranges:
                    symbol_records.extend(_eligible_samples_for_range(group["trading_date"], listing, quality, config, *ranges["valid"], "instrument_test"))
                if "test" in ranges:
                    symbol_records.extend(_eligible_samples_for_range(group["trading_date"], listing, quality, config, *ranges["test"], "double_oos"))
            for item in symbol_records:
                item["symbol"] = symbol
                all_records.append(item)
                sample_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                sample_count[item["split"]] += 1
            series_meta.append({"symbol": symbol, "min_date": str(group["trading_date"].min().date()), "max_date": str(group["trading_date"].max().date()), "rows": len(group), "symbol_role": "holdout" if symbol in holdout_set else "train"})
    assert_split_disjoint(all_records)
    qlib_provider = output / "qlib_provider"
    _write_qlib_compatible_layout(qlib_provider, series_meta, all_dates)
    native_qlib_version = _dump_native_qlib_provider(qlib_provider, config.qlib_source_path, cached)
    try:
        from ..evaluation.leakage import audit_shortcut_leakage
        audit_parts = []
        label_parts = []
        for _symbol, _group, feats, labs in cached:
            audit_parts.append(feats[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32))
            label_parts.append(labs[LABEL_SCHEMA.names].to_numpy(dtype=np.float32))
        leakage_audit = audit_shortcut_leakage(np.concatenate(audit_parts), np.concatenate(label_parts), FEATURE_NAMES, LABEL_SCHEMA.names)
    except Exception as exc:
        leakage_audit = {"passed": False, "violations": [{"type": "audit_error", "message": str(exc)}]}
    dataset_manifest = {
        "schema_version": "technical-dataset.v2", "dataset_id": output.name,
        "source_market_snapshot_id": snapshot.snapshot_id, "source_data_version": snapshot.manifest.get("data_version"),
        "adjustment": snapshot.manifest.get("adjustment"), "frequency": snapshot.manifest.get("frequency"),
        "symbols_hash": symbol_hash(symbols),
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
        "symbol_split": {
            "holdout_ratio": config.holdout_ratio, "seed": config.symbol_split_seed,
            "train_symbols_hash": symbol_hash(train_symbols), "holdout_symbols_hash": symbol_hash(holdout_symbols),
            "train_symbol_count": len(train_symbols), "holdout_symbol_count": len(holdout_symbols),
        },
        "leakage_audit": {"passed": bool(leakage_audit.get("passed")), "violations": leakage_audit.get("violations", [])},
        "split_overlap": 0,
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
        self.split = canonical_split_name(split)
        self.records = [json.loads(line) for line in (self.dataset_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line]
        self.records = [item for item in self.records if canonical_split_name(str(item["split"])) == self.split]
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
