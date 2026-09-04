from __future__ import annotations

import argparse
import json
from pathlib import Path

from .reliability_gate import promote_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a reliability-tested Technical Transformer checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--target", default="ACTIVE", choices=["VALIDATED", "TESTED", "RELIABILITY_PASSED", "ACTIVE", "REJECTED"]
    )
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    manifest = promote_checkpoint(args.checkpoint, args.target, report)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
