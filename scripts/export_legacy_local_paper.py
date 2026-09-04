"""Export legacy local-paper rows without mutating the source database.

This is a one-way, offline migration aid.  It intentionally lives outside the
production package and only opens SQLite files in read-only mode.  Dry-run is
the default; ``--write`` is required to create the requested JSON export.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

TABLES = (
    "paper_state",
    "paper_equity",
    "paper_run",
    "paper_order_v2",
    "paper_fill",
    "paper_position_lot",
    "paper_cash_ledger",
)


def export_legacy_paper(source: str | Path, output: str | Path, *, write: bool = False) -> dict[str, Any]:
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source database does not exist: {source_path}")
    if write and output_path.exists():
        raise FileExistsError(f"refusing to overwrite export: {output_path}")
    uri = f"file:{source_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        payload = {
            "format": "legacy-local-paper-export.v1",
            "source": str(source_path),
            "tables": {
                table: [dict(row) for row in connection.execute(f'SELECT * FROM "{table}"')]
                for table in TABLES
                if table in tables
            },
        }
    if write:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="SQLite database path (opened read-only)")
    parser.add_argument("--output", required=True, help="JSON destination path")
    parser.add_argument("--write", action="store_true", help="write the export; default is dry-run")
    args = parser.parse_args()
    payload = export_legacy_paper(args.source, args.output, write=args.write)
    print(
        json.dumps(
            {"dry_run": not args.write, "table_counts": {k: len(v) for k, v in payload["tables"].items()}},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
