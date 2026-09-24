from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from .common import MODALITIES, PredictionHeads, build_encoders, encode_modalities


class EMTDLFR(nn.Module):
    """Efficient global-local fusion with low-level reconstruction.

    The training engine supplies a complete and a corrupted view.  Their final
    features are aligned with a stop-gradient cosine loss, reproducing the main
    dual-level restoration idea without carrying a full BERT checkpoint.
    """

    def __init__(
        self,
        feature_dims: Mapping[str, int],
        sequence_lengths: Mapping[str, int],
        d_model: int = 64,
        nhead: int = 4,
        temporal_layers: int = 1,
        fusion_layers: int = 2,
        dropout: float = 0.2,
        head_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.encoders = build_encoders(
            feature_dims, sequence_lengths, d_model, nhead, temporal_layers, dropout
        )
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_fusion = nn.TransformerEncoder(
            fusion_layer, num_layers=fusion_layers, norm=nn.LayerNorm(d_model)
        )
        self.local_from_global = nn.ModuleDict(
            {
                name: nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
                for name in MODALITIES
            }
        )
        self.local_norm = nn.ModuleDict({name: nn.LayerNorm(d_model) for name in MODALITIES})
        self.reconstructors = nn.ModuleDict(
            {name: nn.Linear(d_model, int(feature_dims[name])) for name in MODALITIES}
        )
        self.heads = PredictionHeads(3 * d_model, head_hidden, dropout)

    def forward(self, features: Mapping[str, torch.Tensor], masks: Mapping[str, torch.Tensor]):
        encoded = encode_modalities(self.encoders, features, masks)
        global_tokens = torch.stack([encoded[m]["cls"] for m in MODALITIES], dim=1)
        global_tokens = self.global_fusion(global_tokens)
        reconstruction = {}
        for name in MODALITIES:
            local = encoded[name]["tokens"]
            update, _ = self.local_from_global[name](
                local, global_tokens, global_tokens, need_weights=False
            )
            restored = self.local_norm[name](local + update)
            reconstruction[name] = self.reconstructors[name](restored)
        output = self.heads(global_tokens.reshape(global_tokens.shape[0], -1))
        output["reconstruction"] = reconstruction
        return output

