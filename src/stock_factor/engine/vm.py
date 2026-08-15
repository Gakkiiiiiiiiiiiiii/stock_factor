from __future__ import annotations

import logging

import numpy as np

from stock_factor.engine.ops import get_op
from stock_factor.engine.vocab import FEATURES, MAX_FORMULA_TOKENS

LOGGER = logging.getLogger(__name__)


class StackVM:
    def __init__(self, max_tokens: int = MAX_FORMULA_TOKENS) -> None:
        self.max_tokens = max_tokens

    def execute(self, rpn: list[str], features: dict[str, np.ndarray]) -> np.ndarray | None:
        if not rpn or len(rpn) > self.max_tokens:
            return None
        stack: list[np.ndarray] = []
        try:
            with np.errstate(all="ignore"):
                for token in rpn:
                    if token in FEATURES:
                        if token not in features:
                            return None
                        stack.append(np.asarray(features[token], dtype=float))
                        continue
                    operation = get_op(token)
                    if operation is None:
                        return None
                    function, arity = operation
                    if len(stack) < arity:
                        return None
                    arguments = stack[-arity:]
                    del stack[-arity:]
                    stack.append(
                        np.nan_to_num(
                            np.asarray(function(*arguments), dtype=float), nan=np.nan, posinf=np.nan, neginf=np.nan
                        )
                    )
        except Exception as exc:
            LOGGER.debug("factor formula failed %s: %s", rpn, exc)
            return None
        if len(stack) != 1 or np.isnan(stack[0]).all():
            return None
        return stack[0]
