from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from stock_factor.technical_transformer.data.dataset import deterministic_symbol_split  # noqa: E402
from stock_factor.technical_transformer.data.features import build_features  # noqa: E402
from stock_factor.technical_transformer.data.schemas import FEATURE_NAMES, MASK_RECONSTRUCTION_FEATURES  # noqa: E402
from stock_factor.technical_transformer.evaluation.leakage import audit_shortcut_leakage  # noqa: E402
from stock_factor.technical_transformer.evaluation.reliability_gate import (  # noqa: E402
    evaluate_reliability_gate,
    transition_checkpoint_status,
)
from stock_factor.technical_transformer.model.losses import compute_mask_loss  # noqa: E402
from stock_factor.technical_transformer.training.masking import apply_mask  # noqa: E402
from stock_factor.technical_transformer.training.optimizer import TrainingStage, build_optimizer  # noqa: E402
from stock_factor.technical_transformer.training.train import TechnicalTransformerSystem  # noqa: E402


def _frame(rows: int = 240) -> pd.DataFrame:
    close = 10 + np.linspace(0, 3, rows)
    return pd.DataFrame(
        {
            "trading_date": pd.date_range("2022-01-01", periods=rows, freq="B"),
            "symbol": "AAA.SZ",
            "open": close - 0.02,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": 100000.0,
            "amount": close * 100000.0,
            "turnover": 0.01,
            "listing_days": np.arange(rows) + 1,
        }
    )


def test_shortcut_audit_reports_affine_copy() -> None:
    source = np.linspace(-1, 1, 32)
    result = audit_shortcut_leakage(source[:, None], (2 * source + 1)[:, None], ["feature"], ["target"])
    assert result["passed"] is False
    assert result["violations"][0]["affine_fit_r2"] > 0.9999


def test_train_and_valid_share_structured_mask_semantics() -> None:
    x = torch.randn(2, 128, len(FEATURE_NAMES))
    first = apply_mask(x, mode="mixed", seed=123)
    second = apply_mask(x, mode="mixed", seed=123)
    assert torch.equal(first.positions, second.positions)
    assert torch.equal(first.input, second.input)
    assert torch.equal(first.target, x)
    assert torch.all(first.input[first.positions] == 0)


def test_mask_reconstruction_target_shape() -> None:
    x = torch.randn(1, 128, len(FEATURE_NAMES))
    masked = apply_mask(x, mode="day", seed=1)
    model = TechnicalTransformerSystem(
        {
            "input_dim": len(FEATURE_NAMES),
            "hidden_size": 32,
            "layers": 1,
            "heads": 4,
            "ffn_size": 64,
            "embedding_dim": 16,
        }
    )
    output = model(masked.input, mask_positions=masked.positions)
    indices = tuple(FEATURE_NAMES.index(name) for name in MASK_RECONSTRUCTION_FEATURES)
    assert output["mask_prediction"].shape[-1] == len(MASK_RECONSTRUCTION_FEATURES)
    assert torch.isfinite(
        compute_mask_loss(output["mask_prediction"], masked.target, masked.positions, target_indices=indices)
    )


def test_optimizer_has_real_encoder_and_head_lrs() -> None:
    model = TechnicalTransformerSystem(
        {
            "input_dim": len(FEATURE_NAMES),
            "hidden_size": 32,
            "layers": 1,
            "heads": 4,
            "ffn_size": 64,
            "embedding_dim": 16,
        }
    )
    optimizer = build_optimizer(model, TrainingStage("wyckoff_phase_events", 1, 2e-5, 1e-4))
    assert {group["name"]: group["lr"] for group in optimizer.param_groups} == {"encoder": 2e-5, "heads": 1e-4}


def test_symbol_holdout_is_deterministic_and_disjoint() -> None:
    symbols = [f"{index:06d}.SZ" for index in range(20)]
    train_a, holdout_a = deterministic_symbol_split(symbols, holdout_ratio=0.2, seed=7)
    train_b, holdout_b = deterministic_symbol_split(symbols, holdout_ratio=0.2, seed=7)
    assert train_a == train_b and holdout_a == holdout_b
    assert not set(train_a) & set(holdout_a)
    assert len(holdout_a) == 4


def test_missing_turnover_keeps_calendar_and_observation_flag() -> None:
    bars = _frame()
    bars.loc[160, "turnover"] = np.nan
    features = build_features(bars)
    assert len(features) == len(bars)
    assert features.loc[160, "turnover"] == 0.0
    assert features.loc[160, "turnover_observed"] == 0.0
    assert features.loc[160, "quality_mask"] == 1.0


def test_reliability_gate_does_not_pass_without_evidence() -> None:
    result = evaluate_reliability_gate({})
    assert result["status"] == "FAIL"
    with pytest.raises(ValueError):
        transition_checkpoint_status("CANDIDATE", "ACTIVE", gate_status="FAIL")
