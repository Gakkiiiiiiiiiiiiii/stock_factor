"""Explicit exploratory/legacy analysis integrations.

These providers are compatibility readers only and always mark results as
``formal_eligible=False``. Formal application paths use the v5.1 provider.
"""

from stock_factor.adapters.http.providers.content import ExploratoryContentSignalProvider

LegacyContentSignalProvider = ExploratoryContentSignalProvider

__all__ = ["ExploratoryContentSignalProvider", "LegacyContentSignalProvider"]
