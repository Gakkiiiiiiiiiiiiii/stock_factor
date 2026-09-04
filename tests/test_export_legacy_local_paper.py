from __future__ import annotations

import json
import sqlite3
import subprocess
import sys


def test_legacy_export_is_read_only_and_dry_run_by_default(tmp_path):
    source = tmp_path / "paper.db"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE paper_state (account_id TEXT, cash REAL)")
        connection.execute("INSERT INTO paper_state VALUES ('fixture', 100.0)")
        connection.commit()
    output = tmp_path / "export.json"
    command = [
        sys.executable,
        "scripts/export_legacy_local_paper.py",
        "--source",
        str(source),
        "--output",
        str(output),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["dry_run"] is True
    assert not output.exists()
    with sqlite3.connect(source) as connection:
        assert connection.execute("SELECT cash FROM paper_state WHERE account_id='fixture'").fetchone() == (100.0,)

    subprocess.run([*command, "--write"], check=True)
    exported = json.loads(output.read_text(encoding="utf-8"))
    assert exported["tables"]["paper_state"][0]["account_id"] == "fixture"
