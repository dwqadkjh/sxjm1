from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from .common import MODALITIES, PredictionHeads, build_encoders, encode_modalities, sequence_with_cls


class MulTBaseline(nn.Module):
    """Compact MulT-style cross-modal attention baseline for precomputed features."""

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
        self.cross_attention = nn.ModuleDict()
        for target in MODALITIES:
            for source in MODALITIES:
                if target != source:
                    self.cross_attention[f"{target}_from_{source}"] = nn.MultiheadAttention(
                        d_model, nhead, dropout=dropout, batch_first=True
                    )
        self.norms = nn.ModuleDict({m: nn.LayerNorm(d_model) for m in MODALITIES})
        self.heads = PredictionHeads(3 * d_model, head_hidden, dropout)

    def forward(self, features: Mapping[str, torch.Tensor], masks: Mapping[str, torch.Tensor]):
        encoded = encode_modalities(self.encoders, features, masks)
        enhanced = []
        for target in MODALITIES:
            query = encoded[target]["cls"].unsqueeze(1)
            updates = []
            for source in MODALITIES:
                if source == target:
                    continue
                key_value = sequence_with_cls(encoded[source])
                key_padding = ~encoded[source]["attention_valid"]
                update, _ = self.cross_attention[f"{target}_from_{source}"](
                    query, key_value, key_value, key_padding_mask=key_padding, need_weights=False
                )
                updates.append(update.squeeze(1))
            enhanced.append(self.norms[target](encoded[target]["cls"] + sum(updates)))
        output = self.heads(torch.cat(enhanced, dim=-1))
        output["reconstruction"] = {}
        return output

