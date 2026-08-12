import numpy as np
import pytest

from stock_factor.engine.lookback import max_lookback_from_rpn
from stock_factor.engine.ops import get_op
from stock_factor.engine.vm import StackVM


def test_vm_delay_and_cross_section_rank_are_deterministic():
    close = np.array([[10, 11, 12, 13, 14, 15], [20, 19, 18, 17, 16, 15]], dtype=float)
    result = StackVM().execute(["close", "close", "ts_delay_5", "div"], {"close": close})
    assert result[0, 5] == pytest.approx(1.5)
    assert result[1, 5] == pytest.approx(0.75)
    ranks = StackVM().execute(["close", "cs_rank"], {"close": close})
    assert ranks[0, 0] == pytest.approx(0.5)
    assert ranks[1, 0] == pytest.approx(1.0)


def test_new_ops_and_lookback_semantics():
    values = np.array([[1.0, 2.0, 3.0]])
    function, arity = get_op("decay_linear_3")
    assert arity == 1
    assert function(values)[0, 2] == pytest.approx(14 / 6)
    assert max_lookback_from_rpn(["close", "ts_mean_20", "ts_delay_5", "cs_rank"]) == 25
    with pytest.raises(ValueError):
        max_lookback_from_rpn(["close", "lead_5"])


def test_invalid_formula_is_rejected():
    close = np.ones((2, 10))
    assert StackVM().execute(["close", "bogus"], {"close": close}) is None
    assert StackVM().execute(["close", "close"], {"close": close}) is None
