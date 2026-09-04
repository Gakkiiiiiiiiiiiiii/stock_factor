"""Fail-closed runtime profile and Paper Authority configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

from stock_factor.domain.authority import PaperAuthority, RuntimeProfile


class RuntimeConfigurationError(ValueError):
    """Raised when a profile cannot safely construct the application."""


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RuntimeConfig:
    profile: RuntimeProfile
    paper_authority: PaperAuthority
    quant_base_url: str | None
    required_quant_contract: str
    required_quant_checksum: str | None
    required_content_contract: str
    required_content_checksum: str | None
    allow_local_paper: bool

    @classmethod
    def from_env(
        cls,
        *,
        profile: RuntimeProfile | str | None = None,
        paper_authority: PaperAuthority | str | None = None,
        quant_base_url: str | None = None,
        required_quant_contract: str | None = None,
        required_quant_checksum: str | None = None,
        required_content_contract: str | None = None,
        required_content_checksum: str | None = None,
        allow_local_paper: bool | None = None,
    ) -> "RuntimeConfig":
        explicit = {
            "FACTOR_PAPER_AUTHORITY": paper_authority is not None or "FACTOR_PAPER_AUTHORITY" in os.environ,
            "QUANT_SERVICE_URL": quant_base_url is not None or "QUANT_SERVICE_URL" in os.environ,
            "FACTOR_REQUIRED_QUANT_CONTRACT": (
                required_quant_contract is not None or "FACTOR_REQUIRED_QUANT_CONTRACT" in os.environ
            ),
            "FACTOR_REQUIRED_QUANT_CHECKSUM": (
                required_quant_checksum is not None or "FACTOR_REQUIRED_QUANT_CHECKSUM" in os.environ
            ),
            "FACTOR_REQUIRED_CONTENT_CONTRACT": (
                required_content_contract is not None or "FACTOR_REQUIRED_CONTENT_CONTRACT" in os.environ
            ),
            "FACTOR_REQUIRED_CONTENT_CHECKSUM": (
                required_content_checksum is not None or "FACTOR_REQUIRED_CONTENT_CHECKSUM" in os.environ
            ),
            "ALLOW_LOCAL_PAPER": allow_local_paper is not None or "ALLOW_LOCAL_PAPER" in os.environ,
        }
        resolved_profile = cls._profile(
            profile if profile is not None else os.getenv("FACTOR_RUNTIME_PROFILE", RuntimeProfile.DEV)
        )
        resolved_authority = cls._authority(
            paper_authority
            if paper_authority is not None
            else os.getenv("FACTOR_PAPER_AUTHORITY", PaperAuthority.QUANT)
        )
        resolved_url = (quant_base_url if quant_base_url is not None else os.getenv("QUANT_SERVICE_URL")) or None
        resolved_contract = (
            required_quant_contract
            if required_quant_contract is not None
            else os.getenv("FACTOR_REQUIRED_QUANT_CONTRACT", "paper-account.v1")
        )
        resolved_checksum = (
            required_quant_checksum
            if required_quant_checksum is not None
            else os.getenv("FACTOR_REQUIRED_QUANT_CHECKSUM")
        )
        resolved_content_contract = (
            required_content_contract
            if required_content_contract is not None
            else os.getenv("FACTOR_REQUIRED_CONTENT_CONTRACT", "content-factor-signal.v5.1")
        )
        resolved_content_checksum = (
            required_content_checksum
            if required_content_checksum is not None
            else os.getenv("FACTOR_REQUIRED_CONTENT_CHECKSUM")
        )
        resolved_allow_local = (
            _truthy(os.getenv("ALLOW_LOCAL_PAPER")) if allow_local_paper is None else bool(allow_local_paper)
        )
        config = cls(
            profile=resolved_profile,
            paper_authority=resolved_authority,
            quant_base_url=resolved_url.rstrip("/") if resolved_url else None,
            required_quant_contract=resolved_contract,
            required_quant_checksum=resolved_checksum,
            required_content_contract=resolved_content_contract,
            required_content_checksum=resolved_content_checksum,
            allow_local_paper=resolved_allow_local,
        )
        if resolved_profile in {RuntimeProfile.STAGING, RuntimeProfile.PROD}:
            missing = [name for name, present in explicit.items() if not present]
            if missing:
                raise RuntimeConfigurationError("formal runtime configuration must be explicit: " + ", ".join(missing))
        config.validate()
        return config

    def validate(self) -> None:
        formal = self.profile in {RuntimeProfile.STAGING, RuntimeProfile.PROD}
        if formal:
            required = {
                "FACTOR_PAPER_AUTHORITY": self.paper_authority == PaperAuthority.QUANT,
                "QUANT_SERVICE_URL": bool(self.quant_base_url),
                "FACTOR_REQUIRED_QUANT_CONTRACT": self.required_quant_contract == "paper-account.v1",
                "FACTOR_REQUIRED_QUANT_CHECKSUM": bool(self.required_quant_checksum),
                "FACTOR_REQUIRED_CONTENT_CONTRACT": self.required_content_contract == "content-factor-signal.v5.1",
                "FACTOR_REQUIRED_CONTENT_CHECKSUM": bool(self.required_content_checksum),
                "ALLOW_LOCAL_PAPER": not self.allow_local_paper,
            }
            failed = [name for name, valid in required.items() if not valid]
            if failed:
                raise RuntimeConfigurationError(f"formal runtime configuration is incomplete: {', '.join(failed)}")
        if self.paper_authority == PaperAuthority.LOCAL_EXPERIMENTAL:
            if self.profile not in {RuntimeProfile.DEV, RuntimeProfile.TEST}:
                raise RuntimeConfigurationError("local_experimental Paper Authority is limited to dev/test")
            if not self.allow_local_paper:
                raise RuntimeConfigurationError("ALLOW_LOCAL_PAPER=true is required for local_experimental")

    @staticmethod
    def _profile(value: RuntimeProfile | str) -> RuntimeProfile:
        try:
            return value if isinstance(value, RuntimeProfile) else RuntimeProfile(str(value).strip().lower())
        except ValueError as exc:
            raise RuntimeConfigurationError(f"unsupported runtime profile: {value}") from exc

    @staticmethod
    def _authority(value: PaperAuthority | str) -> PaperAuthority:
        try:
            return value if isinstance(value, PaperAuthority) else PaperAuthority(str(value).strip().lower())
        except ValueError as exc:
            raise RuntimeConfigurationError(f"unsupported Paper Authority: {value}") from exc


__all__ = ["RuntimeConfig", "RuntimeConfigurationError"]
