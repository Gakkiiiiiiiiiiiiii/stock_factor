import hashlib
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


class FeedbackModel:
    def __init__(self):
        self.prompts = []

    def complete(self, prompt, system=None, temperature=None):
        self.prompts.append(prompt)
        candidate = (
            {"name": "first", "hypothesis": "initial", "rpn": ["ret", "cs_rank"]}
            if len(self.prompts) == 1
            else {"name": "second", "hypothesis": "feedback revised", "rpn": ["close", "cs_rank"]}
        )
        return json.dumps({"candidates": [candidate]})


def test_model_candidate_generation_is_injected_and_vocab_checked():
    service = FactorMiningService(FixtureMarket(), FixtureContent(), FixtureFactors(), FixtureModel())
    result = service.run({"symbols": [f"6000{index:02d}" for index in range(20)], "use_model": True})
    assert result["factor_count"] == 1
    assert result["factors"][0]["name"] == "model_reversal"


def test_candidate_identity_keeps_each_rpn_and_hash_together():
    service = FactorMiningService(FixtureMarket(), FixtureContent(), FixtureFactors())
    candidates = [
        {"name": "first", "hypothesis": "one", "rpn": ["ret", "cs_rank"]},
        {"name": "second", "hypothesis": "two", "rpn": ["close", "cs_rank"]},
        {"name": "third", "hypothesis": "three", "rpn": ["volume", "cs_rank"]},
    ]
    result = service.run({"symbols": [f"6000{index:02d}" for index in range(20)], "candidates": candidates})
    assert result["factors"]
    assert {(item["name"], tuple(item["rpn"])) for item in result["factors"]} <= {
        (item["name"], tuple(item["rpn"])) for item in candidates
    }
    for factor in result["factors"]:
        assert factor["candidate_hash"] == hashlib.sha256(" ".join(factor["rpn"]).encode()).hexdigest()


def test_multi_round_mining_feeds_evaluation_feedback_into_mutation():
    service = FactorMiningService(FixtureMarket(), FixtureContent(), FixtureFactors())
    result = service.run(
        {
            "symbols": [f"6000{index:02d}" for index in range(20)],
            "rounds": 2,
            "candidates_per_round": 1,
            "candidates": [{"name": "base", "hypothesis": "base", "rpn": ["ret", "ts_mean_5", "neg", "cs_rank"]}],
        }
    )
    assert [item["round"] for item in result["search_rounds"]] == [1, 2]
    assert {item["metrics"]["generation_round"] for item in result["factors"]} == {1, 2}


def test_model_second_round_receives_structured_feedback_and_previous_formula():
    model = FeedbackModel()
    service = FactorMiningService(FixtureMarket(), FixtureContent(), FixtureFactors(), model)
    result = service.run(
        {
            "symbols": [f"6000{index:02d}" for index in range(20)],
            "use_model": True,
            "rounds": 2,
            "candidates_per_round": 1,
        }
    )
    assert len(model.prompts) == 2
    assert "structured feedback" in model.prompts[1]
    assert "candidate_hash" in model.prompts[1]
    assert {item["name"] for item in result["factors"]} == {"first", "second"}


def test_near_identical_factor_outputs_are_kept_once_for_statistics():
    first = {
        "candidate": {"candidate_hash": "a"},
        "values": __import__("numpy").arange(20).reshape(2, 10),
        "preliminary": {"fitness": 1.0},
    }
    duplicate = {
        "candidate": {"candidate_hash": "b"},
        "values": __import__("numpy").arange(20).reshape(2, 10) * 2,
        "preliminary": {"fitness": 0.5},
    }
    kept, rejected = FactorMiningService._correlation_deduplicate([duplicate, first])
    assert [item["candidate"]["candidate_hash"] for item in kept] == ["a"]
    assert rejected == [{"candidate_hash": "b", "representative": "a"}]
