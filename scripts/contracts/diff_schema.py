"""Produce deterministic semantic diffs for JSON/YAML contract schemas."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

_RANGE_KEYS = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
)


def _load(path: str | Path) -> Any:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"schema root must be an object: {path}")
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fields(schema: dict[str, Any], prefix: str = "") -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    properties = schema.get("properties")
    required = set(schema.get("required", [])) if isinstance(schema.get("required", []), list) else set()
    if not isinstance(properties, dict):
        return result
    for name in sorted(properties):
        child = properties[name]
        if not isinstance(child, dict):
            child = {}
        path = f"{prefix}.{name}" if prefix else name
        record = {
            "required": name in required,
            "type": child.get("type"),
            "enum": child.get("enum"),
            "ranges": {key: child[key] for key in _RANGE_KEYS if key in child},
        }
        result[path] = record
        result.update(_fields(child, path))
    items = schema.get("items")
    if isinstance(items, dict):
        result.update(_fields(items, f"{prefix}[]" if prefix else "[]"))
    return result


def _range_breaking(key: str, before: Any, after: Any) -> bool:
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return True
    if key in {"minimum", "exclusiveMinimum", "minLength", "minItems"}:
        return after > before
    return after < before


def semantic_diff(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Compare required/optional/type/enum/range semantics with stable ordering."""

    before = _fields(old)
    after = _fields(new)
    changes: list[dict[str, Any]] = []
    for path in sorted(set(before) | set(after)):
        left, right = before.get(path), after.get(path)
        if left is None:
            changes.append(
                {
                    "path": path,
                    "kind": "field_added",
                    "severity": "breaking" if right["required"] else "non_breaking",
                    "before": None,
                    "after": right,
                }
            )
            continue
        if right is None:
            changes.append(
                {"path": path, "kind": "field_removed", "severity": "breaking", "before": left, "after": None}
            )
            continue
        if left["required"] != right["required"]:
            changes.append(
                {
                    "path": path,
                    "kind": "requiredness_changed",
                    "severity": "breaking" if right["required"] else "non_breaking",
                    "before": left["required"],
                    "after": right["required"],
                }
            )
        if left["type"] != right["type"]:
            changes.append(
                {
                    "path": path,
                    "kind": "type_changed",
                    "severity": "breaking",
                    "before": left["type"],
                    "after": right["type"],
                }
            )
        if left["enum"] != right["enum"]:
            changes.append(
                {
                    "path": path,
                    "kind": "enum_changed",
                    "severity": "breaking",
                    "before": left["enum"],
                    "after": right["enum"],
                }
            )
        for key in sorted(set(left["ranges"]) | set(right["ranges"])):
            old_value, new_value = left["ranges"].get(key), right["ranges"].get(key)
            if old_value != new_value:
                changes.append(
                    {
                        "path": path,
                        "kind": "range_changed",
                        "range": key,
                        "severity": "breaking"
                        if old_value is None or new_value is None or _range_breaking(key, old_value, new_value)
                        else "non_breaking",
                        "before": old_value,
                        "after": new_value,
                    }
                )
    changes.sort(key=lambda item: (item["path"], item["kind"], item.get("range", "")))
    return {"breaking": any(item["severity"] == "breaking" for item in changes), "changes": changes}


def git_semantic_diff(base_ref: str, *, repo_root: str | Path = ".") -> dict[str, Any]:
    """Diff changed checked-in schemas against a git base without network access."""

    root = Path(repo_root).resolve()
    completed = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=AMRT", f"{base_ref}...HEAD", "--", "contracts"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    reports: list[dict[str, Any]] = []
    for relative in sorted(line.strip().replace("\\", "/") for line in completed.stdout.splitlines() if line.strip()):
        if Path(relative).suffix.lower() not in {".json", ".yaml", ".yml"}:
            continue
        current = root / relative
        if not current.is_file():
            continue
        old_bytes = subprocess.run(
            ["git", "show", f"{base_ref}:{relative}"], cwd=root, check=False, capture_output=True
        )
        if old_bytes.returncode != 0:
            # A newly added contract has no previous consumer surface to diff.
            continue
        old = yaml.safe_load(old_bytes.stdout.decode("utf-8"))
        new = _load(current)
        if not isinstance(old, dict):
            raise ValueError(f"schema root must be an object in {base_ref}:{relative}")
        reports.append({"path": relative, "diff": semantic_diff(old, new)})
    reports.sort(key=lambda item: item["path"])
    return {"base_ref": base_ref, "breaking": any(item["diff"]["breaking"] for item in reports), "contracts": reports}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", nargs="?")
    parser.add_argument("new", nargs="?")
    parser.add_argument("--base-ref", help="compare changed contracts to a git base revision")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    try:
        if args.base_ref:
            result = git_semantic_diff(args.base_ref, repo_root=args.repo_root)
        elif args.old and args.new:
            result = semantic_diff(_load(args.old), _load(args.new))
        else:
            raise ValueError("old and new schemas are required unless --base-ref is supplied")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(json.dumps({"breaking": True, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 1 if result["breaking"] else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
