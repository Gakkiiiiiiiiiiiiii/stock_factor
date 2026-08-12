import json

from stock_factor.application.mining import FactorMiningService
from tests.test_integration import FixtureContent, FixtureMarket


class FixtureFactors:
    def save(self, factor):
        return factor.to_dict()


class FixtureModel:
    def complete(self, prompt, system=None, temperature=None):
        return json.dumps(
            {
                "candidates": [
                    {
                        "name": "model_reversal",
                        "hypothesis": "mean reversion",
                        "rpn": ["ret", "ts_mean_5", "neg", "cs_rank"],
                    }
                ]
            }
        )


def test_model_candidate_generation_is_injected_and_vocab_checked():
    service = FactorMiningService(FixtureMarket(), FixtureContent(), FixtureFactors(), FixtureModel())
    result = service.run(
        {"symbols": [f"6000{index:02d}" for index in range(20)], "use_model": True}
    )
    assert result["factor_count"] == 1
    assert result["factors"][0]["name"] == "model_reversal"
