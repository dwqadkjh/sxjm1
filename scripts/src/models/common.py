from __future__ import annotations

from typing import Dict, Mapping

import torch
from torch import nn


MODALITIES = ("text", "audio", "vision")


class ModalityEncoder(nn.Module):
    """Small temporal Transformer that remains stable when a modality is fully missing."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        max_len: int,
        nhead: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = nn.Parameter(torch.randn(1, max_len + 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, observed_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, length, _ = x.shape
        if length + 1 > self.position.shape[1]:
            raise ValueError(f"Sequence length {length} exceeds configured maximum {self.position.shape[1] - 1}")
        tokens = self.input_proj(x)
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.dropout(tokens + self.position[:, : length + 1])
        cls_valid = torch.ones(batch, 1, dtype=torch.bool, device=x.device)
        attention_valid = torch.cat([cls_valid, observed_mask.bool()], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~attention_valid)
        return {
            "cls": encoded[:, 0],
            "tokens": encoded[:, 1:],
            "attention_valid": attention_valid,
        }


class PredictionHeads(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.regression = nn.Linear(hidden_dim, 1)
        self.classification = nn.Linear(hidden_dim, 3)

    def forward(self, fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.shared(fused)
        return {
            "feature": hidden,
            "regression": self.regression(hidden).squeeze(-1),
            "classification": self.classification(hidden),
        }


def build_encoders(
    feature_dims: Mapping[str, int],
    sequence_lengths: Mapping[str, int],
    d_model: int,
    nhead: int,
    layers: int,
    dropout: float,
) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            name: ModalityEncoder(
                input_dim=int(feature_dims[name]),
                d_model=d_model,
                max_len=int(sequence_lengths[name]),
                nhead=nhead,
                layers=layers,
                dropout=dropout,
            )
            for name in MODALITIES
        }
    )


def encode_modalities(
    encoders: nn.ModuleDict,
    features: Mapping[str, torch.Tensor],
    observed_masks: Mapping[str, torch.Tensor],
) -> Dict[str, Dict[str, torch.Tensor]]:
    return {name: encoders[name](features[name], observed_masks[name]) for name in MODALITIES}


def sequence_with_cls(encoded: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([encoded["cls"].unsqueeze(1), encoded["tokens"]], dim=1)

