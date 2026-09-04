from __future__ import annotations

import torch

from stock_factor.technical_transformer.data.schemas import LABEL_SCHEMA
from stock_factor.technical_transformer.evaluation.baseline_runner import _baseline_loss


def test_baseline_loss_phase_slice_supports_backward() -> None:
    prediction = torch.randn(4, len(LABEL_SCHEMA.names), requires_grad=True)
    target = torch.zeros_like(prediction)
    phase = LABEL_SCHEMA.slices["phase"]
    target[:, phase] = torch.softmax(torch.randn(4, len(range(phase.start, phase.stop))), dim=-1)
    valid = torch.ones_like(target, dtype=torch.bool)
    loss = _baseline_loss(prediction, target, valid)
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
