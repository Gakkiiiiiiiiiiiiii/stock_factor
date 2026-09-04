"""Factor mining application package.

The public import remains ``application.mining.FactorMiningService`` while
candidate generation and screening live in dedicated modules.
"""

from stock_factor.application.mining.generate import generate_candidates, model_candidates, mutate_candidate
from stock_factor.application.mining.screen import canonical_candidates, correlation_deduplicate, feedback
from stock_factor.application.mining.service import FactorMiningService, ResearchIntegrityError

__all__ = [
    "FactorMiningService",
    "ResearchIntegrityError",
    "canonical_candidates",
    "correlation_deduplicate",
    "feedback",
    "generate_candidates",
    "model_candidates",
    "mutate_candidate",
]
