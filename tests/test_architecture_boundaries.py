"""架构依赖规则测试（设计文档 §6.2）。

stock_factor 只允许通过 HTTP 依赖 quant / stock_content，
禁止任何 Python import 形式的跨仓依赖。
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "stock_factor"
FORBIDDEN_PREFIXES = ("stock_agent", "stock_content", "quant_demo", "quant")


def _iter_imports(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_factor_does_not_import_other_repositories():
    violations: list[str] = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for module in _iter_imports(tree):
            if module == "quant" or module.startswith(tuple(prefix + "." for prefix in FORBIDDEN_PREFIXES)):
                violations.append(f"{path.relative_to(ROOT)}: {module}")
    assert violations == [], f"stock_factor 违反依赖规则（§6.2）: {violations}"


def test_market_data_provider_points_to_quant_by_default(monkeypatch):
    from stock_factor.adapters.http.providers import HttpMarketDataProvider

    monkeypatch.delenv("MARKET_DATA_SERVICE_URL", raising=False)
    provider = HttpMarketDataProvider()
    # §12/§76：默认事实源必须是 quant，而不是 stock-agent-market-data
    assert provider._url == "http://quant:8011"  # noqa: SLF001
