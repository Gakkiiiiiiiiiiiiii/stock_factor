from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_checkpoint_manifest(checkpoint_dir: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_dir) / "checkpoint_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_status(checkpoint_dir: str | Path) -> str:
    return str(read_checkpoint_manifest(checkpoint_dir).get("checkpoint_status", "UNKNOWN"))
