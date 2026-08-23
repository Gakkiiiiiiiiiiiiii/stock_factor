from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

FEATURE_SCHEMA_VERSION = "technical-feature.v2"
LABEL_SCHEMA_VERSION = "technical-label.v2"


# These three features are algebraically equivalent to MA targets and are
# intentionally excluded from V2 to remove the shortcut path.
CONTINUOUS_FEATURES = [
    "ret_1", "ret_3", "ret_5", "ret_10",
    "open_prev_close", "high_prev_close", "low_prev_close", "close_prev_close",
    "intraday_range_prev_close", "intraday_body_prev_close", "body_ratio",
    "upper_shadow_ratio", "lower_shadow_ratio", "range_position", "gap_ratio",
    "log1p_volume", "volume_ratio_5", "volume_ratio_10", "volume_ratio_20", "volume_ratio_60",
    "volume_zscore_20", "volume_zscore_60", "volume_change_1", "volume_change_5",
    "amount_ratio_5", "amount_ratio_20", "turnover", "turnover_ratio_5",
    "turnover_ratio_20", "turnover_zscore_20", "true_range_close", "atr14_close",
    "realized_vol_5", "realized_vol_20", "realized_vol_60", "range_atr14",
]

STATE_FEATURES = [
    "is_suspended", "is_st", "is_star_st", "is_delisting",
    "is_limit_up", "is_limit_down",
    "st_observed", "suspension_observed", "limit_observed", "delisting_observed",
    "turnover_observed", "price_observed", "volume_observed",
    "listing_days_norm", "quality_mask",
]

FEATURE_NAMES = CONTINUOUS_FEATURES + STATE_FEATURES

MA_LABELS = [
    "ma5_slope", "ma10_slope", "ma20_slope", "ma30_slope", "ma60_slope", "ma120_slope",
    "close_ma5_distance", "close_ma20_distance", "close_ma60_distance", "close_ma120_distance",
    "bull_alignment_score", "bear_alignment_score", "ma_trend_strength",
    "compression_score", "ma_expansion_score",
    "bull_cross_5_20", "bear_cross_5_20", "bull_cross_10_20", "bear_cross_10_20",
    "bull_cross_20_60", "bear_cross_20_60",
    "days_since_bull_cross_5_20", "days_since_bear_cross_5_20",
    "days_since_bull_cross_20_60", "days_since_bear_cross_20_60",
]

BOLL_LABELS = [
    "percent_b", "boll_zscore", "bandwidth", "bandwidth_delta_5", "bandwidth_delta_20",
    "squeeze_score", "boll_expansion_score", "upper_break_strength", "lower_break_strength",
    "bandwidth_percentile",
]

WYCKOFF_PRIMITIVE_LABELS = [
    "trend_direction", "wyckoff_trend_strength", "trading_range_score", "range_position",
    "support_distance", "resistance_distance", "breakout_strength", "breakdown_strength",
    "false_breakout_score", "volume_expansion", "volume_contraction", "effort_result_score",
    "effort_result_divergence", "demand_pressure_proxy", "supply_pressure_proxy",
]

PHASE_LABELS = ["accumulation_like", "markup", "distribution_like", "markdown", "transition"]
EVENT_LABELS = ["sc_score", "bc_score", "spring_score", "upthrust_score", "sos_score", "sow_score"]
ALL_LABELS = MA_LABELS + BOLL_LABELS + WYCKOFF_PRIMITIVE_LABELS + PHASE_LABELS + EVENT_LABELS

MASK_RECONSTRUCTION_FEATURES = [
    "ret_1", "intraday_range_prev_close", "intraday_body_prev_close", "range_position",
    "log1p_volume", "volume_ratio_20", "turnover", "amount_ratio_20", "atr14_close",
    "realized_vol_20",
]

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "PRICE": (
        "ret_1", "ret_3", "ret_5", "ret_10", "open_prev_close", "high_prev_close",
        "low_prev_close", "close_prev_close", "intraday_range_prev_close",
        "intraday_body_prev_close", "body_ratio", "upper_shadow_ratio", "lower_shadow_ratio",
        "range_position", "gap_ratio",
    ),
    "VOLUME": (
        "log1p_volume", "volume_ratio_5", "volume_ratio_10", "volume_ratio_20", "volume_ratio_60",
        "volume_zscore_20", "volume_zscore_60", "volume_change_1", "volume_change_5",
        "amount_ratio_5", "amount_ratio_20", "turnover", "turnover_ratio_5",
        "turnover_ratio_20", "turnover_zscore_20",
    ),
    "VOLATILITY": (
        "true_range_close", "atr14_close", "realized_vol_5", "realized_vol_20",
        "realized_vol_60", "range_atr14",
    ),
}


@dataclass(frozen=True)
class LabelSpec:
    name: str
    group: str
    task_type: str
    role: str = "CORE_SEQUENCE"


def _build_label_specs() -> tuple[LabelSpec, ...]:
    specs: list[LabelSpec] = []
    for name in MA_LABELS:
        task_type = "binary_event" if name.startswith(("bull_cross_", "bear_cross_")) else "regression"
        specs.append(LabelSpec(name, "ma", task_type, "CORE_SEQUENCE"))
    specs.extend(LabelSpec(name, "bollinger", "regression", "CORE_SEQUENCE") for name in BOLL_LABELS)
    auxiliary = {"volume_expansion", "volume_contraction", "demand_pressure_proxy", "supply_pressure_proxy"}
    for name in WYCKOFF_PRIMITIVE_LABELS:
        specs.append(LabelSpec(name, "wyckoff_primitives", "regression", "AUXILIARY" if name in auxiliary else "CORE_SEQUENCE"))
    specs.extend(LabelSpec(name, "phase", "soft_distribution", "CORE_SEQUENCE") for name in PHASE_LABELS)
    specs.extend(LabelSpec(name, "events", "binary_event", "CORE_SEQUENCE") for name in EVENT_LABELS)
    return tuple(specs)


LABEL_SPECS = _build_label_specs()
LABEL_SPEC_BY_NAME = {item.name: item for item in LABEL_SPECS}


@dataclass(frozen=True)
class LabelSchema:
    version: str = LABEL_SCHEMA_VERSION
    ma: tuple[str, ...] = tuple(MA_LABELS)
    bollinger: tuple[str, ...] = tuple(BOLL_LABELS)
    wyckoff_primitives: tuple[str, ...] = tuple(WYCKOFF_PRIMITIVE_LABELS)
    phase: tuple[str, ...] = tuple(PHASE_LABELS)
    events: tuple[str, ...] = tuple(EVENT_LABELS)

    @property
    def specs(self) -> tuple[LabelSpec, ...]:
        return LABEL_SPECS

    def spec_for(self, name: str) -> LabelSpec:
        try:
            return LABEL_SPEC_BY_NAME[name]
        except KeyError as exc:
            raise KeyError(f"unknown label: {name}") from exc

    @property
    def names(self) -> list[str]:
        return list(self.ma + self.bollinger + self.wyckoff_primitives + self.phase + self.events)

    @property
    def slices(self) -> dict[str, slice]:
        start = 0
        result: dict[str, slice] = {}
        for name, values in (
            ("ma", self.ma), ("bollinger", self.bollinger),
            ("wyckoff_primitives", self.wyckoff_primitives),
            ("phase", self.phase), ("events", self.events),
        ):
            result[name] = slice(start, start + len(values))
            start += len(values)
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.version,
            "groups": {
                name: list(values) for name, values in (
                    ("ma", self.ma), ("bollinger", self.bollinger),
                    ("wyckoff_primitives", self.wyckoff_primitives),
                    ("phase", self.phase), ("events", self.events),
                )
            },
            "specs": [
                {"name": item.name, "group": item.group, "task_type": item.task_type, "role": item.role}
                for item in self.specs
            ],
        }


LABEL_SCHEMA = LabelSchema()


def schema_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


FEATURE_SCHEMA = {
    "schema_version": FEATURE_SCHEMA_VERSION,
    "dimensions": len(FEATURE_NAMES),
    "continuous_features": CONTINUOUS_FEATURES,
    "state_features": STATE_FEATURES,
    "feature_groups": {name: list(values) for name, values in FEATURE_GROUPS.items()},
    "mask_reconstruction_features": MASK_RECONSTRUCTION_FEATURES,
    "normalization": {"continuous": "train_only_robust_clip", "state": "none", "mask": "learned_embedding"},
}
