from __future__ import annotations

from stock_factor.engine.ops import parse_ts_token
from stock_factor.engine.vocab import BINARY_OPS, CS_OPS, FEATURES, TERNARY_OPS, TS_BINARY_OPS, UNARY_OPS

FORBIDDEN_FUTURE_TOKENS = ("lead", "future_return", "negative_delay", "centered_rolling")


def _pop(stack: list[int], token: str) -> int:
    if not stack:
        raise ValueError(f"not enough operands for token: {token}")
    return stack.pop()


def max_lookback_from_rpn(rpn: list[str]) -> int:
    stack: list[int] = []
    for token in map(str, rpn):
        if any(marker in token for marker in FORBIDDEN_FUTURE_TOKENS):
            raise ValueError(f"future-looking token is forbidden: {token}")
        parsed = parse_ts_token(token)
        if token in FEATURES:
            stack.append(1)
        elif parsed:
            name, window = parsed
            values = [_pop(stack, token) for _ in range(2 if name in TS_BINARY_OPS else 1)]
            stack.append(max(values) + (window if name in {"ts_delay", "ts_delta"} else window - 1))
        elif token in CS_OPS or token in UNARY_OPS:
            stack.append(_pop(stack, token))
        elif token in BINARY_OPS:
            stack.append(max(_pop(stack, token), _pop(stack, token)))
        elif token in TERNARY_OPS:
            stack.append(max(_pop(stack, token), _pop(stack, token), _pop(stack, token)))
        else:
            raise ValueError(f"unknown factor token: {token}")
    if len(stack) != 1:
        raise ValueError("invalid RPN expression")
    return max(1, stack[0])
