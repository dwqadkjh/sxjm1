"""Language-dominated noise-resistant learning for precomputed features.

Reference: Zhang, Wang and Yu, NeurIPS 2024.  This compact adaptation preserves
the two central ideas of LNLN: dominant-language correction through an
audio/visual proxy, and language-guided multimodal fusion.  It uses the common
project mask protocol and dual task heads required by this project.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .common import (
    MODALITIES,
    PredictionHeads,
    build_encoders,
    encode_modalities,
    sequence_with_cls,
)


class LNLNAdapted(nn.Module):
    """Language-dominated correction plus hyper-modality-style fusion."""

    requires_valid_masks = True

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
        self.proxy_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.proxy_fallback = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.proxy_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.proxy_norm = nn.LayerNorm(d_model)
        self.completeness_head = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.audio_to_language = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.vision_to_language = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.hyper_norm = nn.LayerNorm(d_model)
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            fusion_layer,
            num_layers=fusion_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.reconstructors = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(4 * d_model, 2 * d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(2 * d_model, int(feature_dims[name])),
                )
                for name in MODALITIES
            }
        )
        self.heads = PredictionHeads(3 * d_model, head_hidden, dropout)

    @staticmethod
    def _key_padding(encoded: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return ~encoded["attention_valid"].bool()

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
        valid_masks: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        encoded = encode_modalities(self.encoders, features, masks)
        batch = features["text"].shape[0]

        observed_ratio = masks["text"].float().sum(dim=1) / valid_masks[
            "text"
        ].float().sum(dim=1).clamp_min(1.0)
        language = encoded["text"]["cls"]
        completeness = self.completeness_head(
            torch.cat([language, observed_ratio.unsqueeze(-1)], dim=-1)
        ).squeeze(-1)

        audio_available = masks["audio"].any(dim=1)
        vision_available = masks["vision"].any(dim=1)
        fallback = self.proxy_fallback.expand(batch, -1, -1)
        proxy_sources = torch.cat(
            [
                encoded["audio"]["cls"].unsqueeze(1),
                encoded["vision"]["cls"].unsqueeze(1),
                fallback,
            ],
            dim=1,
        )
        source_padding = torch.stack(
            [~audio_available, ~vision_available, torch.zeros_like(audio_available)], dim=1
        )
        query = self.proxy_query.expand(batch, -1, -1)
        proxy, _ = self.proxy_attention(
            query,
            proxy_sources,
            proxy_sources,
            key_padding_mask=source_padding,
            need_weights=False,
        )
        proxy = self.proxy_norm(proxy.squeeze(1) + query.squeeze(1))
        corrected_language = (
            completeness.unsqueeze(-1) * language
            + (1.0 - completeness).unsqueeze(-1) * proxy
        )

        language_query = corrected_language.unsqueeze(1)
        audio_sequence = sequence_with_cls(encoded["audio"])
        vision_sequence = sequence_with_cls(encoded["vision"])
        audio_update, _ = self.audio_to_language(
            language_query,
            audio_sequence,
            audio_sequence,
            key_padding_mask=self._key_padding(encoded["audio"]),
            need_weights=False,
        )
        vision_update, _ = self.vision_to_language(
            language_query,
            vision_sequence,
            vision_sequence,
            key_padding_mask=self._key_padding(encoded["vision"]),
            need_weights=False,
        )
        hyper = self.hyper_norm(
            corrected_language + audio_update.squeeze(1) + vision_update.squeeze(1)
        )
        global_tokens = torch.stack(
            [hyper, encoded["audio"]["cls"], encoded["vision"]["cls"]], dim=1
        )
        global_tokens = self.fusion(global_tokens)

        global_context = torch.cat(
            [corrected_language, encoded["audio"]["cls"], encoded["vision"]["cls"]],
            dim=-1,
        )
        reconstruction = {}
        for name in MODALITIES:
            local = encoded[name]["tokens"]
            context = global_context.unsqueeze(1).expand(-1, local.shape[1], -1)
            reconstruction[name] = self.reconstructors[name](
                torch.cat([local, context], dim=-1)
            )

        output = self.heads(global_tokens.flatten(start_dim=1))
        output["reconstruction"] = reconstruction
        output["completeness"] = completeness
        output["proxy_language"] = proxy
        output["aux_losses"] = {
            "lnln_completeness": F.mse_loss(completeness, observed_ratio)
        }
        return output


__all__ = ["LNLNAdapted"]
