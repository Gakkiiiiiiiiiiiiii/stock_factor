from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from stock_factor.adapters.http.quant_paper_client import QuantPaperClient
from stock_factor.api.dependencies import build_application
from stock_factor.api.main import create_app
from stock_factor.config.runtime import RuntimeConfig, RuntimeConfigurationError
from stock_factor.domain.authority import PaperAuthority, RuntimeProfile
from tests.test_integration import FixtureContent, FixtureMarket


def test_default_runtime_is_quant_and_not_local(monkeypatch):
    for name in (
        "FACTOR_RUNTIME_PROFILE",
        "FACTOR_PAPER_AUTHORITY",
        "QUANT_SERVICE_URL",
        "MARKET_DATA_SERVICE_URL",
        "FACTOR_REQUIRED_QUANT_CHECKSUM",
        "ALLOW_LOCAL_PAPER",
    ):
        monkeypatch.delenv(name, raising=False)
    config = RuntimeConfig.from_env()
    assert config.profile is RuntimeProfile.DEV
    assert config.paper_authority is PaperAuthority.QUANT
    assert config.allow_local_paper is False


@pytest.mark.parametrize("missing", ["authority", "url", "contract", "checksum", "allow"])
def test_formal_profiles_require_explicit_quant_requirements(monkeypatch, missing):
    monkeypatch.setenv("FACTOR_RUNTIME_PROFILE", "prod")
    monkeypatch.setenv("FACTOR_PAPER_AUTHORITY", "quant")
    monkeypatch.setenv("QUANT_SERVICE_URL", "http://quant")
    monkeypatch.setenv("FACTOR_REQUIRED_QUANT_CONTRACT", "paper-account.v1")
    monkeypatch.setenv("FACTOR_REQUIRED_QUANT_CHECKSUM", "sha256:fixture")
    monkeypatch.setenv("FACTOR_REQUIRED_CONTENT_CONTRACT", "content-factor-signal.v5.1")
    monkeypatch.setenv("FACTOR_REQUIRED_CONTENT_CHECKSUM", "sha256:content")
    monkeypatch.setenv("ALLOW_LOCAL_PAPER", "false")
    env_name = {
        "authority": "FACTOR_PAPER_AUTHORITY",
        "url": "QUANT_SERVICE_URL",
        "contract": "FACTOR_REQUIRED_QUANT_CONTRACT",
        "checksum": "FACTOR_REQUIRED_QUANT_CHECKSUM",
        "allow": "ALLOW_LOCAL_PAPER",
    }[missing]
    monkeypatch.delenv(env_name, raising=False)
    with pytest.raises(RuntimeConfigurationError):
        RuntimeConfig.from_env()


def test_formal_profile_rejects_empty_contract_even_when_other_settings_exist(monkeypatch):
    monkeypatch.setenv("FACTOR_RUNTIME_PROFILE", "prod")
    monkeypatch.setenv("FACTOR_PAPER_AUTHORITY", "quant")
    monkeypatch.setenv("QUANT_SERVICE_URL", "http://quant")
    monkeypatch.setenv("FACTOR_REQUIRED_QUANT_CONTRACT", "")
    monkeypatch.setenv("FACTOR_REQUIRED_QUANT_CHECKSUM", "sha256:fixture")
    monkeypatch.setenv("FACTOR_REQUIRED_CONTENT_CONTRACT", "content-factor-signal.v5.1")
    monkeypatch.setenv("FACTOR_REQUIRED_CONTENT_CHECKSUM", "sha256:content")
    monkeypatch.setenv("ALLOW_LOCAL_PAPER", "false")
    with pytest.raises(RuntimeConfigurationError):
        RuntimeConfig.from_env()


@pytest.mark.parametrize(
    ("authority", "allow_local"),
    [("local_experimental", "false"), ("quant", "true")],
)
def test_formal_profile_rejects_local_or_local_opt_in(monkeypatch, authority, allow_local):
    monkeypatch.setenv("FACTOR_RUNTIME_PROFILE", "staging")
    monkeypatch.setenv("FACTOR_PAPER_AUTHORITY", authority)
    monkeypatch.setenv("QUANT_SERVICE_URL", "http://quant")
    monkeypatch.setenv("FACTOR_REQUIRED_QUANT_CONTRACT", "paper-account.v1")
    monkeypatch.setenv("FACTOR_REQUIRED_QUANT_CHECKSUM", "sha256:fixture")
    monkeypatch.setenv("FACTOR_REQUIRED_CONTENT_CONTRACT", "content-factor-signal.v5.1")
    monkeypatch.setenv("FACTOR_REQUIRED_CONTENT_CHECKSUM", "sha256:content")
    monkeypatch.setenv("ALLOW_LOCAL_PAPER", allow_local)
    with pytest.raises(RuntimeConfigurationError):
        RuntimeConfig.from_env()


def test_local_paper_requires_explicit_authority_and_opt_in(monkeypatch):
    monkeypatch.setenv("FACTOR_RUNTIME_PROFILE", "test")
    monkeypatch.setenv("FACTOR_PAPER_AUTHORITY", "local_experimental")
    monkeypatch.delenv("ALLOW_LOCAL_PAPER", raising=False)
    with pytest.raises(RuntimeConfigurationError):
        RuntimeConfig.from_env()
    monkeypatch.setenv("ALLOW_LOCAL_PAPER", "true")
    config = RuntimeConfig.from_env()
    assert config.paper_authority is PaperAuthority.LOCAL_EXPERIMENTAL


def test_default_composition_uses_quant_proxy_without_local_repository(tmp_path, monkeypatch):
    monkeypatch.delenv("FACTOR_PAPER_AUTHORITY", raising=False)
    monkeypatch.delenv("ALLOW_LOCAL_PAPER", raising=False)
    application = build_application(f"sqlite:///{tmp_path / 'authority.db'}", FixtureMarket(), FixtureContent())
    assert isinstance(application._paper, QuantPaperClient)  # noqa: SLF001


def test_quant_failure_returns_503_without_local_takeover(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTOR_RUNTIME_PROFILE", "test")
    monkeypatch.setenv("FACTOR_PAPER_AUTHORITY", "quant")
    monkeypatch.setenv("QUANT_SERVICE_URL", "http://quant")
    monkeypatch.setattr(httpx, "request", lambda *args, **kwargs: (_ for _ in ()).throw(httpx.ConnectError("down")))
    application = build_application(f"sqlite:///{tmp_path / 'quant-failure.db'}", FixtureMarket(), FixtureContent())
    response = TestClient(create_app(application)).get("/api/v1/paper/state")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "QUANT_UNAVAILABLE"
    assert isinstance(application._paper, QuantPaperClient)  # noqa: SLF001


def test_research_ready_reports_authority_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("FACTOR_RUNTIME_PROFILE", "staging")
    monkeypatch.setenv("FACTOR_PAPER_AUTHORITY", "quant")
    monkeypatch.setenv("QUANT_SERVICE_URL", "http://quant")
    monkeypatch.setenv("FACTOR_REQUIRED_QUANT_CONTRACT", "paper-account.v1")
    monkeypatch.setenv("FACTOR_REQUIRED_QUANT_CHECKSUM", "sha256:fixture")
    monkeypatch.setenv("FACTOR_REQUIRED_CONTENT_CONTRACT", "content-factor-signal.v5.1")
    monkeypatch.setenv("FACTOR_REQUIRED_CONTENT_CHECKSUM", "sha256:content")
    monkeypatch.setenv("ALLOW_LOCAL_PAPER", "false")
    response = TestClient(
        create_app(build_application(f"sqlite:///{tmp_path / 'ready.db'}", FixtureMarket(), FixtureContent()))
    ).get("/health/research-ready")
    assert response.status_code == 200
    payload = response.json()
    assert payload["runtime_profile"] == "staging"
    assert payload["paper_authority"] == "quant"
    assert payload["required_quant_checksum_configured"] is True


def test_research_ready_is_not_formal_without_capability_and_does_not_drift(monkeypatch, tmp_path):
    monkeypatch.setenv("FACTOR_RUNTIME_PROFILE", "dev")
    monkeypatch.setenv("FACTOR_PAPER_AUTHORITY", "quant")
    monkeypatch.delenv("QUANT_SERVICE_URL", raising=False)
    monkeypatch.delenv("FACTOR_REQUIRED_QUANT_CHECKSUM", raising=False)
    monkeypatch.setenv("ALLOW_LOCAL_PAPER", "false")
    application = build_application(f"sqlite:///{tmp_path / 'not-ready.db'}", FixtureMarket(), FixtureContent())
    client = TestClient(create_app(application))
    monkeypatch.setenv("QUANT_SERVICE_URL", "http://quant")
    monkeypatch.setenv("FACTOR_REQUIRED_QUANT_CHECKSUM", "sha256:late")
    payload = client.get("/health/research-ready").json()
    assert payload["status"] == "not_ready"
    assert payload["formal_eligible"] is False
    assert payload["quant_base_url_configured"] is False
    assert payload["required_quant_checksum_configured"] is False
