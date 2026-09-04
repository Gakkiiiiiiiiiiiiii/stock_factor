"""Architecture checks for the P2-01 application/provider split."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "stock_factor"


def test_application_use_cases_are_packages_with_thin_compatibility_shims():
    for package in (
        SRC / "application" / "mining",
        SRC / "application" / "oos",
        SRC / "application" / "artifacts",
        SRC / "application" / "promotion",
    ):
        assert (package / "__init__.py").is_file()
    for shim in (
        SRC / "application" / "final_oos_evaluation.py",
        SRC / "application" / "oos_authorization.py",
        SRC / "application" / "research_artifact_service.py",
        SRC / "application" / "factor_set_service.py",
    ):
        lines = shim.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 5
        assert "deprecated" in "\n".join(lines).lower()


def test_domain_modules_have_no_framework_or_external_boundary_imports():
    forbidden = {"fastapi", "httpx", "sqlalchemy", "torch"}
    violations = []
    for path in (SRC / "domain").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = node.module if isinstance(node, ast.ImportFrom) else None
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                modules = [module] if module else []
            violations.extend((path.name, name) for name in modules if name.split(".")[0] in forbidden)
    assert violations == []


def test_http_provider_modules_are_separated_and_legacy_facade_is_small():
    providers = SRC / "adapters" / "http"
    for module in (
        providers / "providers" / "market.py",
        providers / "providers" / "content.py",
        providers / "providers" / "model.py",
    ):
        assert module.is_file()
    assert not (providers / "providers.py").exists()


def test_mining_orchestrator_delegates_use_cases_to_split_modules():
    source = (SRC / "application" / "mining" / "service.py").read_text(encoding="utf-8")
    for symbol in (
        "generate_candidates",
        "canonical_candidates",
        "cohort_statistics",
        "schedule_final_oos",
        "assemble_research_artifact",
        "evaluate_candidate_promotion",
        "promote_candidates",
    ):
        assert symbol in source or symbol in (SRC / "application" / "promotion" / "candidates.py").read_text(
            encoding="utf-8"
        )
