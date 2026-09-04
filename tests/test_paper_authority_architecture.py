from pathlib import Path


def test_production_source_has_no_local_paper_write_implementation():
    root = Path(__file__).parents[1]
    source = root / "src" / "stock_factor"
    assert not (source / "application" / "paper.py").exists()
    assert not (source / "experimental" / "local_paper").exists()
    for path in (source / "api" / "dependencies.py", source / "application" / "service.py"):
        assert "application.paper" not in path.read_text(encoding="utf-8")
        assert "local_paper" not in path.read_text(encoding="utf-8")
    production_text = "\n".join(path.read_text(encoding="utf-8") for path in source.rglob("*.py"))
    for forbidden in ("PostgresPaperRepository", "PaperStateRow", "PaperTradingService", "PaperRepository"):
        assert forbidden not in production_text
