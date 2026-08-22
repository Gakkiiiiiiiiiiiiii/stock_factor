from __future__ import annotations

import torch
from torch import nn

from ..data.schemas import CONTINUOUS_FEATURES, FEATURE_NAMES, STATE_FEATURES


class TechnicalTransformer(nn.Module):
    """V1 128-day encoder with a learnable CLS representation."""

    def __init__(
        self,
        input_dim: int = len(FEATURE_NAMES),
        hidden_size: int = 384,
        layers: int = 6,
        heads: int = 8,
        ffn_size: int = 1536,
        dropout: float = 0.10,
        embedding_dim: int = 256,
        sequence_length: int = 128,
    ) -> None:
        super().__init__()
        if input_dim != len(FEATURE_NAMES):
            raise ValueError(f"V1 expects {len(FEATURE_NAMES)} features, got {input_dim}")
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.continuous_indices = tuple(FEATURE_NAMES.index(name) for name in CONTINUOUS_FEATURES)
        self.state_indices = tuple(FEATURE_NAMES.index(name) for name in STATE_FEATURES)
        self.continuous_projection = nn.Linear(len(CONTINUOUS_FEATURES), hidden_size)
        self.state_projection = nn.Linear(len(STATE_FEATURES), hidden_size)
        self.position = nn.Parameter(torch.zeros(1, sequence_length + 1, hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=heads, dim_feedforward=ffn_size,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(hidden_size)
        self.embedding_projection = nn.Sequential(nn.Linear(hidden_size, embedding_dim), nn.LayerNorm(embedding_dim))
        self.mask_head = nn.Linear(hidden_size, 6)

    def forward(self, x: torch.Tensor, *, padding_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != self.sequence_length or x.shape[2] != len(FEATURE_NAMES):
            raise ValueError(f"expected [batch,{self.sequence_length},{len(FEATURE_NAMES)}], got {tuple(x.shape)}")
        continuous = x[..., self.continuous_indices]
        state = x[..., self.state_indices]
        day = self.continuous_projection(continuous) + self.state_projection(state)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, day], dim=1) + self.position
        mask = None
        if padding_mask is not None:
            mask = torch.cat([torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device), padding_mask], dim=1)
        hidden = self.encoder(tokens, src_key_padding_mask=mask)
        hidden = self.final_norm(hidden)
        cls_hidden = hidden[:, 0]
        return {
            "token_hidden": hidden[:, 1:], "cls_hidden": cls_hidden,
            "technical_embedding": self.embedding_projection(cls_hidden), "mask_prediction": self.mask_head(hidden[:, 1:]),
        }
