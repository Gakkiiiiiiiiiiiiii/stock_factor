from __future__ import annotations

FEATURES = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover",
    "vwap",
    "ret",
    "event_heat",
    "theme_sentiment",
    "video_bullish_claim_count",
    "video_bearish_claim_count",
    "verified_catalyst_count",
    "verified_risk_count",
    "content_attention_score",
    "author_attention_score",
    "cross_video_consensus",
    "cross_video_disagreement",
)
TS_WINDOWS = (3, 4, 5, 8, 10, 15, 20, 30, 60, 120)
TS_OPS = (
    "ts_mean",
    "ts_std",
    "ts_max",
    "ts_min",
    "ts_delta",
    "ts_delay",
    "ts_rank",
    "ts_sum",
    "decay_linear",
    "ts_argmax",
    "ts_argmin",
    "count",
)
TS_BINARY_OPS = ("ts_corr", "ts_cov")
CS_OPS = ("cs_rank", "cs_zscore", "cs_demean")
UNARY_OPS = ("neg", "abs", "log", "sqrt", "sign", "signedpower")
BINARY_OPS = ("add", "sub", "mul", "div", "gt", "lt", "max", "min")
TERNARY_OPS = ("where",)
FEATURE_SET = frozenset(FEATURES)
ALL_OP_TOKENS = frozenset(
    [f"{name}_{window}" for name in TS_OPS + TS_BINARY_OPS for window in TS_WINDOWS]
    + list(CS_OPS)
    + list(UNARY_OPS)
    + list(BINARY_OPS)
    + list(TERNARY_OPS)
)
MAX_FORMULA_TOKENS = 16


def is_valid_token(token: str) -> bool:
    return token in FEATURE_SET or token in ALL_OP_TOKENS
