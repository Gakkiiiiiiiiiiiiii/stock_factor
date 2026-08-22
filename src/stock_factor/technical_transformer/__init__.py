"""Technical Transformer V1.

This package is deliberately independent from the market-data authority.  It
only consumes an immutable snapshot produced by ``quant`` and writes research
dataset/checkpoint manifests that retain the full lineage.
"""

__all__ = ["__version__"]

__version__ = "technical-transformer.v1"
