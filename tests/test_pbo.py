"""PBO（详细修改方案 P1-3）。"""
from __future__ import annotations

import numpy as np

from stock_factor.engine.pbo import pbo_report


def test_pbo_high_for_pure_noise_candidates():
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 0.01, size=(20, 64))
    report = pbo_report(noise, partitions=8)
    assert report["passed"] is False or report["pbo"] > 0.3
    assert report["trials"] == 20 and report["periods"] == 64


def test_pbo_low_for_persistent_signal():
    rng = np.random.default_rng(9)
    # 持续强信号（均值 0.01 远大于噪声标准误），IS 选择能稳定命中信号 trial
    base = rng.normal(0.01, 0.001, size=(1, 64))
    signal = base + rng.normal(0.0, 0.0002, size=(5, 64)) + np.linspace(0, 0.001, 5)[:, None]
    noise = rng.normal(0.0, 0.01, size=(15, 64))
    returns = np.vstack([signal, noise])
    report = pbo_report(returns, partitions=8, max_pbo=0.35)
    assert report["pbo"] < 0.35


def test_pbo_insufficient_input_is_worst_case():
    report = pbo_report(np.zeros((1, 2)))
    assert report["pbo"] == 1.0 and report["passed"] is False
