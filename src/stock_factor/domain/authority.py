"""Runtime authority values shared by composition roots and readiness checks."""

from __future__ import annotations

from enum import StrEnum


class RuntimeProfile(StrEnum):
    DEV = "dev"
    TEST = "test"
    STAGING = "staging"
    PROD = "prod"


class PaperAuthority(StrEnum):
    QUANT = "quant"
    LOCAL_EXPERIMENTAL = "local_experimental"


__all__ = ["PaperAuthority", "RuntimeProfile"]
