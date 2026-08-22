from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_factor.technical_transformer.data.snapshot import QuantSnapshot, write_quant_snapshot  # noqa: E402


def _date_column(frame: pd.DataFrame, name: str) -> pd.Series:
    values = frame[name].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    return pd.to_datetime(values, format="%Y%m%d", errors="coerce")


def prepare(args: argparse.Namespace) -> QuantSnapshot:
    quant_root = Path(args.quant_root)
    history_path = Path(args.history_path or quant_root / "data/parquet/user_pattern_history.parquet")
    capital_path = Path(args.capital_path or quant_root / "data/parquet/joinquant_microcap_capital.parquet")
    st_path = Path(args.st_path or quant_root / "data/parquet/joinquant_microcap_st_status.parquet")
    bars = pd.read_parquet(history_path)
    bars["trading_date"] = pd.to_datetime(bars["trading_date"]).dt.normalize()
    bars = bars.sort_values(["symbol", "trading_date"]).reset_index(drop=True)
    if bars.duplicated(["symbol", "trading_date"]).any():
        raise ValueError("history contains duplicate symbol/trading_date rows")
    capital = pd.read_parquet(capital_path)
    capital["announce_date"] = _date_column(capital, "announce_date")
    capital = capital.dropna(subset=["announce_date", "circulating_capital"])
    capital = capital[capital["circulating_capital"].astype(float) > 0].sort_values(["announce_date", "symbol"])
    bars = pd.merge_asof(
        bars.sort_values(["trading_date", "symbol"]),
        capital[["symbol", "announce_date", "circulating_capital"]].sort_values(["announce_date", "symbol"]),
        left_on="trading_date", right_on="announce_date", by="symbol", direction="backward",
    )
    bars["turnover"] = bars["volume"].astype(float) / bars["circulating_capital"].astype(float)
    active_capital_missing = bars["circulating_capital"].isna() & (bars["volume"].astype(float) > 0)
    excluded_active_missing_turnover = int(active_capital_missing.sum())
    # A missing PIT float-share fact cannot be replaced by a future value.
    # Drop only active rows; suspension rows remain as masked tokens with a
    # zero turnover observation.
    bars = bars.loc[~active_capital_missing].copy()
    bars["turnover"] = bars["turnover"].fillna(0.0)
    bars = bars.drop(columns=["announce_date", "circulating_capital"])
    if st_path.exists():
        status = pd.read_parquet(st_path)
        status["trading_date"] = pd.to_datetime(status["trading_date"]).dt.normalize()
        status = status[["symbol", "trading_date", "is_st_history"]].drop_duplicates(["symbol", "trading_date"])
        bars = bars.merge(status, on=["symbol", "trading_date"], how="left")
        bars["is_st"] = bars["is_st_history"].fillna(False).astype(float)
        bars = bars.drop(columns=["is_st_history"])
    else:
        bars["is_st"] = 0.0
    bars["is_suspended"] = ((bars["volume"].astype(float) <= 0) | bars["close"].isna()).astype(float)
    bars["is_star_st"] = 0.0
    bars["is_delisting"] = 0.0
    bars["is_limit_up"] = 0.0
    bars["is_limit_down"] = 0.0
    bars["state_observed"] = 1.0 if st_path.exists() else 0.0
    bars["listing_days"] = bars.groupby("symbol").cumcount() + 1
    suspended = bars["is_suspended"] > 0
    price_present = bars[["open", "high", "low", "close"]].notna().all(axis=1)
    ohlc_checks = (
        (suspended | price_present)
        & (~price_present | (bars["high"] >= bars[["open", "close"]].max(axis=1)))
        & (~price_present | (bars["low"] <= bars[["open", "close"]].min(axis=1)))
        & (~price_present | (bars["high"] >= bars["low"]))
    )
    checks = (
        ohlc_checks
        & (bars["volume"] >= 0)
        & (bars["amount"] >= 0)
        & (bars["turnover"] >= 0)
    )
    if not checks.all():
        raise ValueError(f"OHLC/volume/turnover quality check failed for {(~checks).sum()} rows")
    if bars["turnover"].isna().any():
        raise ValueError(f"PIT circulating capital missing for {bars['turnover'].isna().sum()} rows")
    source_data_version = hashlib.sha256(
        (str(history_path.resolve()) + str(history_path.stat().st_size) + str(history_path.stat().st_mtime_ns)).encode()
    ).hexdigest()
    snapshot_root = Path(args.snapshot_root or quant_root / "data/market_snapshots")
    return write_quant_snapshot(
        bars,
        snapshot_root,
        source="quant",
        source_data_version=source_data_version,
        adjustment="qfq",
        frequency="1d",
        snapshot_id=args.snapshot_id,
        extra_manifest={
            "turnover_semantics": "volume / PIT circulating_capital",
            "turnover_source": str(capital_path.resolve()),
            "pit_sources": {"st": str(st_path.resolve()), "capital": str(capital_path.resolve())},
            "quality_flags": [],
            "excluded_active_missing_turnover_rows": excluded_active_missing_turnover,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze quant history into a verified Technical Transformer snapshot")
    parser.add_argument("--quant-root", default=r"D:\project\quant")
    parser.add_argument("--history-path")
    parser.add_argument("--capital-path")
    parser.add_argument("--st-path")
    parser.add_argument("--snapshot-root")
    parser.add_argument("--snapshot-id")
    args = parser.parse_args()
    snapshot = prepare(args)
    print(json.dumps(snapshot.verify(), ensure_ascii=False, indent=2))
    print(json.dumps(snapshot.manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
