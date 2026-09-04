from __future__ import annotations

import torch
from torch import nn


class LastDayMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_size), nn.GELU(), nn.Linear(hidden_size, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x[:, -1])


class GRUBaseline(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.head(hidden[-1])


class TCNBaseline(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_dim, hidden_size, kernel_size=3, padding=2, dilation=1),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=4, dilation=2),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.network(x.transpose(1, 2))[:, :, -1])


class TransformerBaseline(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, hidden_size: int = 128, layers: int = 2, heads: int = 4
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_size)
        block = nn.TransformerEncoderLayer(hidden_size, heads, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.head = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(self.projection(x))[:, -1])


def make_baseline(name: str, input_dim: int, output_dim: int, **kwargs) -> nn.Module:
    values = {
        "mlp": LastDayMLP,
        "last_day_mlp": LastDayMLP,
        "gru": GRUBaseline,
        "tcn": TCNBaseline,
        "transformer": TransformerBaseline,
    }
    try:
        return values[name.lower()](input_dim, output_dim, **kwargs)
    except KeyError as exc:
        raise ValueError(f"unknown baseline: {name}") from exc


def relative_gain(transformer_score: float, baseline_score: float) -> float:
    return float(transformer_score / max(abs(baseline_score), 1e-12) - 1.0)


def compare_baselines(
    scores: dict[str, float], *, transformer_name: str = "transformer", baseline_name: str = "gru"
) -> dict[str, float | bool]:
    if transformer_name not in scores or baseline_name not in scores:
        raise ValueError("both transformer and baseline scores are required")
    gain = relative_gain(scores[transformer_name], scores[baseline_name])
    return {
        "transformer_score": float(scores[transformer_name]),
        "baseline_score": float(scores[baseline_name]),
        "relative_gain": gain,
        "passes_five_percent_gain": gain >= 0.05,
    }
