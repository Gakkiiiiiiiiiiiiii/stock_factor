import pytest

from stock_factor.api.dependencies import build_application
from stock_factor.config.runtime import RuntimeConfigurationError
from tests.test_integration import FixtureContent, FixtureMarket


def test_local_paper_configuration_is_rejected_by_production_composition(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTOR_RUNTIME_PROFILE", "test")
    monkeypatch.setenv("FACTOR_PAPER_AUTHORITY", "local_experimental")
    monkeypatch.setenv("ALLOW_LOCAL_PAPER", "true")
    with pytest.raises(RuntimeConfigurationError, match="Quant Paper Authority"):
        build_application(f"sqlite:///{tmp_path / 'paper.db'}", FixtureMarket(), FixtureContent())
