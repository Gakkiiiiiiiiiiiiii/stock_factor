from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluator import evaluate_all_splits
from .report import build_reliability_report, write_reliability_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen Technical Transformer checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gold-set")
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    checkpoint_dir = Path(args.checkpoint)
    dataset_dir = Path(args.dataset)
    identity = json.loads((checkpoint_dir / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    splits = evaluate_all_splits(checkpoint_dir, dataset_dir, device=args.device, batch_size=args.batch_size)
    gold = {"status": "NOT_PROVIDED"}
    if args.gold_set:
        gold = {"status": "LOADED", "path": str(Path(args.gold_set).resolve()), "evaluation": "REQUIRES_HUMAN_LABEL_MATCH"}
    report = build_reliability_report(checkpoint_identity=identity, dataset_manifest=manifest, splits=splits, gold_set=gold)
    json_path, markdown_path = write_reliability_report(report, args.report)
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path), "gate": report["reliability_gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
