"""MISA-style shared/private representation model for precomputed features.

Reference: Hazarika et al., ACM MM 2020.  This is a compact adaptation of the
official MISA design to the project's masked ``[B,T,D]`` feature interface.  It
keeps the shared/private decomposition, orthogonality, CMD similarity,
reconstruction, six-token Transformer fusion, and dual prediction heads.
"""

from __future__ import annotations

from itertools import combinations
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .common import MODALITIES, PredictionHeads, build_encoders, encode_modalities


def _difference_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Penalize correlation between two representation spaces."""
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    x = F.normalize(x, p=2, dim=0, eps=1e-8)
    y = F.normalize(y, p=2, dim=0, eps=1e-8)
    return (x.transpose(0, 1) @ y).pow(2).mean()


def _cmd_loss(x: torch.Tensor, y: torch.Tensor, moments: int = 3) -> torch.Tensor:
    """Central moment discrepancy used to align modality-shared codes."""
    mx, my = x.mean(dim=0), y.mean(dim=0)
    sx, sy = x - mx, y - my
    loss = torch.linalg.vector_norm(mx - my)
    for order in range(2, moments + 1):
        loss = loss + torch.linalg.vector_norm(
            sx.pow(order).mean(dim=0) - sy.pow(order).mean(dim=0)
        )
    return loss / float(moments)


class MISAAdapted(nn.Module):
    """Modality-invariant/specific fusion adapted to masked feature sequences."""

    def __init__(
        self,
        feature_dims: Mapping[str, int],
        sequence_lengths: Mapping[str, int],
        d_model: int = 64,
        nhead: int = 4,
        temporal_layers: int = 1,
        fusion_layers: int = 1,
        dropout: float = 0.2,
        head_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.encoders = build_encoders(
            feature_dims, sequence_lengths, d_model, nhead, temporal_layers, dropout
        )
        self.private = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
                for name in MODALITIES
            }
        )
        self.shared = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.reconstructors = nn.ModuleDict(
            {name: nn.Linear(d_model, d_model) for name in MODALITIES}
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            layer,
            num_layers=fusion_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.heads = PredictionHeads(6 * d_model, head_hidden, dropout)

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        encoded = encode_modalities(self.encoders, features, masks)
        original = {name: encoded[name]["cls"] for name in MODALITIES}
        private = {name: self.private[name](original[name]) for name in MODALITIES}
        shared = {name: self.shared(original[name]) for name in MODALITIES}

        reconstruction_loss = torch.stack(
            [
                F.mse_loss(
                    self.reconstructors[name](private[name] + shared[name]),
                    original[name],
                )
                for name in MODALITIES
            ]
        ).mean()
        difference_terms = [
            _difference_loss(private[name], shared[name]) for name in MODALITIES
        ]
        difference_terms.extend(
            _difference_loss(private[left], private[right])
            for left, right in combinations(MODALITIES, 2)
        )
        similarity_terms = [
            _cmd_loss(shared[left], shared[right])
            for left, right in combinations(MODALITIES, 2)
        ]

        tokens = torch.stack(
            [*(private[name] for name in MODALITIES), *(shared[name] for name in MODALITIES)],
            dim=1,
        )
        tokens = self.fusion(tokens)
        output = self.heads(tokens.flatten(start_dim=1))
        output["reconstruction"] = {}
        output["aux_losses"] = {
            "misa_reconstruction": reconstruction_loss,
            "misa_difference": torch.stack(difference_terms).mean(),
            "misa_similarity": torch.stack(similarity_terms).mean(),
        }
        return output


__all__ = ["MISAAdapted"]
