"""Immutable research artifact application services."""

from stock_factor.application.artifacts.assemble import assemble_research_artifact
from stock_factor.application.artifacts.seal import *  # noqa: F403

__all__ = ["assemble_research_artifact"]
