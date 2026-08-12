import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"


def test_factor_core_only_depends_on_ports():
    imports = set()
    for source in ROOT.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
            elif isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
    assert not {name for name in imports if name.startswith("stock_agent") or name.startswith("stock_content")}


def test_engine_has_no_api_database_or_http_imports():
    forbidden = ("fastapi", "sqlalchemy", "httpx")
    violations = []
    for source in (ROOT / "stock_factor" / "engine").rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            elif isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
        violations.extend(f"{source.name}: {name}" for name in names if name.startswith(forbidden))
    assert not violations
