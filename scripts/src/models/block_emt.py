"""连续局部缺失专用的 Block-aware EMT；设计借鉴 EMT-DLFR，但不是原论文实现。

本程序及代码是在人工智能工具辅助下完成的；提交前请按竞赛规定补充实际
工具名称、版本/型号、开发机构与版本发布日期，并由参赛队独立核查。
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from .common import MODALITIES, PredictionHeads, build_encoders, encode_modalities


def gap_geometry(observed: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """[B,T,4]: 可观测、距左/右已观测位置、当前连续缺失段长度（均已归一化）。"""
    observed = observed.bool() & valid.bool()
    valid = valid.bool()
    batch, length = observed.shape
    t = torch.arange(length, device=observed.device).expand(batch, -1)
    valid_length = valid.long().sum(dim=1, keepdim=True)
    left = torch.cummax(torch.where(observed, t, -1), dim=1).values
    right = torch.flip(
        torch.cummin(
            torch.flip(torch.where(observed, t, valid_length), dims=(1,)), dim=1
        ).values,
        dims=(1,),
    )
    missing = valid & ~observed
    scale = max(length, 1)
    return torch.stack(
        [
            observed.float(),
            torch.where(missing, (t - left).float() / scale, 0.0),
            torch.where(missing, (right - t).float() / scale, 0.0),
            torch.where(missing, (right - left - 1).float() / scale, 0.0),
        ],
        dim=-1,
    ) * valid.unsqueeze(-1)


class BlockAwareEMT(nn.Module):
    """EMT 全局—局部思想 + 连续缺失几何 + 逐格可观测门控 + 潜空间恢复。

    仅适用三模态时序位置相互对应的 aligned_50 版本；不接受 unaligned_50。
    """

    requires_valid_masks = True

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
        learn_task_uncertainty: bool = False,
        use_modality_auxiliary: bool = False,
        global_context_mode: str = "legacy",
    ) -> None:
        super().__init__()
        if len({int(sequence_lengths[m]) for m in MODALITIES}) != 1:
            raise ValueError("block_emt 需要 aligned_50 的三模态同长度时序")
        self.encoders = build_encoders(
            feature_dims, sequence_lengths, d_model, nhead, temporal_layers, dropout
        )
        self.geometry_proj = nn.ModuleDict(
            {m: nn.Linear(4, d_model, bias=False) for m in MODALITIES}
        )
        self.modality_embedding = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)
        self.global_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        if global_context_mode not in {"legacy", "gated", "pooled_only"}:
            raise ValueError(
                "global_context_mode must be legacy, gated, or pooled_only"
            )
        self.global_context_mode = global_context_mode
        fusion_layer = nn.TransformerEncoderLayer(
            d_model, nhead, 4 * d_model, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.global_fusion = nn.TransformerEncoder(
            fusion_layer, num_layers=fusion_layers, norm=nn.LayerNorm(d_model)
        )
        self.reliability = nn.Sequential(
            nn.Linear(d_model * 2 + 4, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )
        self.time_position = nn.Parameter(
            torch.randn(1, int(sequence_lengths["text"]), d_model) * 0.02
        )
        self.time_norm = nn.LayerNorm(d_model)
        time_layer = nn.TransformerEncoderLayer(
            d_model, nhead, 4 * d_model, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.time_fusion = nn.TransformerEncoder(
            time_layer, num_layers=1, norm=nn.LayerNorm(d_model)
        )
        self.reconstructors = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(d_model * 2 + 4, d_model), nn.GELU(),
                    nn.LayerNorm(d_model),
                )
                for m in MODALITIES
            }
        )
        self.heads = PredictionHeads(d_model * 2, head_hidden, dropout)
        self.modality_heads = (
            nn.ModuleDict(
                {m: PredictionHeads(d_model, head_hidden, dropout) for m in MODALITIES}
            )
            if use_modality_auxiliary else None
        )
        if learn_task_uncertainty:
            self.task_log_vars = nn.Parameter(torch.zeros(2))  # classification, regression
        else:
            self.register_parameter("task_log_vars", None)
        # 新门控放在旧模块之后初始化，使legacy/gated消融共享模块的初始权重一致。
        self.global_reliability = (
            nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
            )
            if global_context_mode != "legacy" else None
        )

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        observed_masks: Mapping[str, torch.Tensor],
        valid_masks: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if valid_masks is None:
            raise ValueError("block_emt 需要 valid_masks，以区分 padding 与局部缺失")
        lengths = {features[m].shape[1] for m in MODALITIES}
        if len(lengths) != 1:
            raise ValueError("block_emt 输入必须是 aligned_50 同长度时序")
        batch, length, _ = features["text"].shape
        observed = {
            m: observed_masks[m].bool() & valid_masks[m].bool() for m in MODALITIES
        }
        # 即便调用者忘记把缺失段置零，也不会读到被遮挡的原始数值。
        safe_features = {
            m: torch.where(observed[m].unsqueeze(-1), features[m], 0.0)
            for m in MODALITIES
        }
        encoded = encode_modalities(self.encoders, safe_features, observed)
        geometry = {
            m: gap_geometry(observed[m], valid_masks[m]) for m in MODALITIES
        }
        local = {
            m: encoded[m]["tokens"] + self.geometry_proj[m](geometry[m])
            for m in MODALITIES
        }

        # EMT 风格全局信息瓶颈：一个融合查询只读取真正有观测值的模态摘要。
        available = torch.stack([observed[m].any(dim=1) for m in MODALITIES], dim=1)
        modality_tokens = (
            torch.stack([encoded[m]["cls"] for m in MODALITIES], dim=1)
            + self.modality_embedding
        )
        global_tokens = torch.cat(
            [self.global_query.expand(batch, -1, -1),
             modality_tokens], dim=1
        )
        global_valid = torch.cat(
            [torch.ones(batch, 1, device=available.device, dtype=torch.bool), available], dim=1
        )
        legacy_context = self.global_fusion(
            global_tokens, src_key_padding_mask=~global_valid
        )[:, 0]
        global_scores = (
            self.global_reliability(modality_tokens).squeeze(-1)
            if self.global_reliability is not None
            else torch.zeros_like(available, dtype=modality_tokens.dtype)
        )
        global_scores = global_scores.masked_fill(~available, -1e4)
        global_weights = torch.softmax(global_scores, dim=1) * available
        global_weights = global_weights / global_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        gated_context = (global_weights.unsqueeze(-1) * modality_tokens).sum(dim=1)
        context = (
            legacy_context if self.global_context_mode == "legacy" else gated_context
        )

        # 门控权重只能分给该时间格实际观测到的模态；全缺失格权重严格为零。
        local_stack = torch.stack([local[m] for m in MODALITIES], dim=2)
        geometry_stack = torch.stack([geometry[m] for m in MODALITIES], dim=2)
        observed_stack = torch.stack([observed[m] for m in MODALITIES], dim=2)
        context_stack = context[:, None, None, :].expand(-1, length, 3, -1)
        scores = self.reliability(
            torch.cat([local_stack, context_stack, geometry_stack], dim=-1)
        ).squeeze(-1)
        scores = scores.masked_fill(~observed_stack, -1e4)
        weights = torch.softmax(scores, dim=2) * observed_stack
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
        mixed = (weights.unsqueeze(-1) * local_stack).sum(dim=2)

        time_valid = torch.stack([valid_masks[m].bool() for m in MODALITIES], dim=2).any(dim=2)
        timeline = self.time_norm(
            mixed + context[:, None, :] + self.time_position[:, :length]
        )
        timeline = self.time_fusion(timeline, src_key_padding_mask=~time_valid)
        timeline = timeline * time_valid.unsqueeze(-1)
        pooled = timeline.sum(dim=1) / time_valid.sum(dim=1, keepdim=True).clamp_min(1)
        head_context = (
            torch.zeros_like(context)
            if self.global_context_mode == "pooled_only"
            else context
        )
        output = self.heads(torch.cat([head_context, pooled], dim=-1))
        output["latent_reconstruction"] = {
            m: self.reconstructors[m](
                torch.cat([timeline, local[m], geometry[m]], dim=-1)
            )
            for m in MODALITIES
        }
        output["latent_targets"] = {m: local[m] for m in MODALITIES}
        output["reliability_weights"] = weights
        output["global_reliability_weights"] = global_weights
        available_distribution = observed_stack.float()
        available_distribution = available_distribution / available_distribution.sum(
            dim=2, keepdim=True
        ).clamp_min(1.0)
        flat_valid = observed_stack.any(dim=2).unsqueeze(-1)
        gate_mean = (weights * flat_valid).sum(dim=(0, 1))
        gate_mean = gate_mean / flat_valid.sum().clamp_min(1.0)
        target_mean = (available_distribution * flat_valid).sum(dim=(0, 1))
        target_mean = target_mean / flat_valid.sum().clamp_min(1.0)
        local_gate_balance = (
                gate_mean.clamp_min(1e-8)
                * (gate_mean.clamp_min(1e-8).log() - target_mean.clamp_min(1e-8).log())
            ).sum()
        global_target = available.float()
        global_target = global_target / global_target.sum(dim=1, keepdim=True).clamp_min(1.0)
        global_mean = global_weights.mean(dim=0)
        global_target_mean = global_target.mean(dim=0)
        global_gate_balance = (
            global_mean.clamp_min(1e-8)
            * (
                global_mean.clamp_min(1e-8).log()
                - global_target_mean.clamp_min(1e-8).log()
            )
        ).sum()
        output["aux_losses"] = {
            "gate_balance": 0.5 * (local_gate_balance + global_gate_balance)
        }
        if self.modality_heads is not None:
            output["modality_auxiliary"] = {
                m: self.modality_heads[m](encoded[m]["cls"]) for m in MODALITIES
            }
            output["modality_available"] = available
        if self.task_log_vars is not None:
            output["task_log_vars"] = self.task_log_vars
        return output
