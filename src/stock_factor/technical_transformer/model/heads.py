from __future__ import annotations

import torch
from torch import nn

from ..data.schemas import LABEL_SCHEMA


class TechnicalHeads(nn.Module):
    def __init__(self, hidden_size: int = 384) -> None:
        super().__init__()
        self.ma = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, len(LABEL_SCHEMA.ma)))
        self.bollinger = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, len(LABEL_SCHEMA.bollinger)))
        self.wyckoff_primitives = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, len(LABEL_SCHEMA.wyckoff_primitives))
        )
        self.phase = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, len(LABEL_SCHEMA.phase)))
        self.events = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, len(LABEL_SCHEMA.events)))

    def forward(self, cls_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "ma": self.ma(cls_hidden),
            "bollinger": self.bollinger(cls_hidden),
            "wyckoff_primitives": self.wyckoff_primitives(cls_hidden),
            "phase": self.phase(cls_hidden),
            "events": self.events(cls_hidden),
        }
