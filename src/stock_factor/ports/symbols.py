"""Canonical cross-service security identifiers."""

from __future__ import annotations

import re

_CN = re.compile(r"^(?:CN\.A\.)?(\d{6})(?:\.(?:SH|SZ|BJ))?$")
_HK = re.compile(r"^(?:HK\.)?(\d{1,5})(?:\.HK)?$", re.IGNORECASE)


def normalize_symbol(value: str) -> str:
    """Return the canonical contract form (CN.A.600519 / HK.00700 / US.NVDA)."""
    raw = str(value or "").strip().upper()
    if match := _CN.fullmatch(raw):
        return f"CN.A.{match.group(1)}"
    if match := _HK.fullmatch(raw):
        return f"HK.{match.group(1).zfill(5)}"
    if raw.startswith("US."):
        return raw
    if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,15}", raw):
        return f"US.{raw}"
    raise ValueError(f"unsupported symbol format: {value!r}")
