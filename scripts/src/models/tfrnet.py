from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from .common import MODALITIES, PredictionHeads, build_encoders, encode_modalities


class TFRNet(nn.Module):
    """Feature-reconstruction network adapted to continuous local missing blocks."""

    def __init__(
        self,
        feature_dims: Mapping[str, int],
        sequence_lengths: Mapping[str, int],
        d_model: int = 64,
        nhead: int = 4,
        temporal_layers: int = 1,
        dropout: float = 0.2,
        head_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.encoders = build_encoders(
            feature_dims, sequence_lengths, d_model, nhead, temporal_layers, dropout
        )
        self.reconstructors = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(3 * d_model, 2 * d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(2 * d_model, int(feature_dims[name])),
                )
                for name in MODALITIES
            }
        )
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=1,
            norm=nn.LayerNorm(d_model),
        )
        self.heads = PredictionHeads(3 * d_model, head_hidden, dropout)

    def forward(self, features: Mapping[str, torch.Tensor], masks: Mapping[str, torch.Tensor]):
        encoded = encode_modalities(self.encoders, features, masks)
        globals_ = torch.stack([encoded[m]["cls"] for m in MODALITIES], dim=1)
        globals_ = self.fusion(globals_)
        reconstruction = {}
        for i, name in enumerate(MODALITIES):
            other = [globals_[:, j] for j in range(len(MODALITIES)) if j != i]
            context = torch.cat(other, dim=-1).unsqueeze(1).expand(-1, encoded[name]["tokens"].shape[1], -1)
            reconstruction[name] = self.reconstructors[name](
                torch.cat([encoded[name]["tokens"], context], dim=-1)
            )
        output = self.heads(globals_.reshape(globals_.shape[0], -1))
        output["reconstruction"] = reconstruction
        return output

