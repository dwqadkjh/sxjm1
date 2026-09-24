from __future__ import annotations

from typing import Any, Mapping

from .emtdlfr import EMTDLFR
from .block_emt import BlockAwareEMT
from .lnln import LNLNAdapted
from .misa import MISAAdapted
from .mult import MulTBaseline
from .tfrnet import TFRNet


def build_model(
    name: str,
    feature_dims: Mapping[str, int],
    sequence_lengths: Mapping[str, int],
    cfg: Mapping[str, Any],
):
    common = dict(
        feature_dims=feature_dims,
        sequence_lengths=sequence_lengths,
        d_model=int(cfg.get("d_model", 64)),
        nhead=int(cfg.get("nhead", 4)),
        temporal_layers=int(cfg.get("temporal_layers", 1)),
        dropout=float(cfg.get("dropout", 0.2)),
        head_hidden=int(cfg.get("head_hidden", 128)),
    )
    key = name.lower().replace("-", "_")
    if key == "mult":
        return MulTBaseline(**common)
    if key in {"tfr", "tfr_net", "tfrnet"}:
        return TFRNet(**common)
    if key in {"emt", "emt_dlfr", "emtdlfr"}:
        return EMTDLFR(**common, fusion_layers=int(cfg.get("fusion_layers", 2)))
    if key in {"block_emt", "blockaware_emt", "block_emt_dlfr", "br_emt_dlfr"}:
        return BlockAwareEMT(
            **common,
            fusion_layers=int(cfg.get("fusion_layers", 1)),
            learn_task_uncertainty=bool(cfg.get("learn_task_uncertainty", False)),
            use_modality_auxiliary=bool(cfg.get("use_modality_auxiliary", False)),
            global_context_mode=str(cfg.get("global_context_mode", "legacy")),
        )
    if key in {"misa", "misa_adapted"}:
        return MISAAdapted(**common, fusion_layers=int(cfg.get("fusion_layers", 1)))
    if key in {"lnln", "lnln_adapted"}:
        return LNLNAdapted(**common, fusion_layers=int(cfg.get("fusion_layers", 2)))
    raise ValueError(
        f"Unknown model {name!r}; choose mult, tfr_net, emt_dlfr, block_emt, misa, or lnln"
    )


__all__ = [
    "build_model", "MulTBaseline", "TFRNet", "EMTDLFR", "BlockAwareEMT",
    "MISAAdapted", "LNLNAdapted",
]
