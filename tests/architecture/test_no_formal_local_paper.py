"""Formal API/application code must not statically import experimental Paper."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src" / "stock_factor"


def test_formal_api_and_application_have_no_experimental_imports():
    violations: list[str] = []
    for package in (ROOT / "api", ROOT / "application"):
        for path in package.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                if any(
                    name == "stock_factor.experimental" or name.startswith("stock_factor.experimental.")
                    for name in names
                ):
                    violations.append(str(path.relative_to(ROOT)))
    assert violations == []
