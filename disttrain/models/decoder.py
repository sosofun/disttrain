from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from disttrain.config import StageConfig, TrainingConfig
from disttrain.models.base import StageModel, TensorDict
from disttrain.models.modalities import build_decoder_modality
from disttrain.models.tp_layers import ColumnParallelLinear
from disttrain.models.tp_transformer import TPTransformerBlock


class DecoderModel(StageModel):
    stage_name = "decoder"

    def __init__(
        self,
        stage_cfg: StageConfig,
        train_cfg: TrainingConfig,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        super().__init__()
        self.output_modalities = list(stage_cfg.output_modalities)
        self.use_activation_checkpoint = stage_cfg.activation_checkpoint
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.text_head = ColumnParallelLinear(
            train_cfg.hidden_size,
            train_cfg.vocab_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            gather_output=True,
        )
        self.blocks = nn.ModuleList(
            [
                TPTransformerBlock(
                    hidden_size=train_cfg.hidden_size,
                    num_heads=train_cfg.num_attention_heads,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    causal=False,
                )
            ]
        )
        self.norm = nn.LayerNorm(train_cfg.hidden_size)

        heads: Dict[str, nn.Module] = {}
        if "image" in self.output_modalities:
            heads["image"] = build_decoder_modality(
                "image",
                hidden_size=train_cfg.hidden_size,
                image_size=train_cfg.image_size,
                tp_size=tp_size,
                tp_rank=tp_rank,
            )
        if "audio" in self.output_modalities:
            heads["audio"] = build_decoder_modality(
                "audio",
                hidden_size=train_cfg.hidden_size,
                audio_length=train_cfg.audio_length,
                tp_size=tp_size,
                tp_rank=tp_rank,
            )
        self.heads = nn.ModuleDict(heads)

    def forward(
        self, inputs: TensorDict, meta: Optional[Dict[str, object]] = None
    ) -> TensorDict:
        if "hidden_states" not in inputs:
            raise KeyError("DecoderModel expects 'hidden_states'")
        hidden = inputs["hidden_states"]
        for block in self.blocks:
            if self.use_activation_checkpoint and self.training:
                hidden = checkpoint.checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        hidden = self.norm(hidden)
        pooled = hidden.mean(dim=1)

        if self.use_activation_checkpoint and self.training:
            text_logits = checkpoint.checkpoint(self.text_head, hidden, use_reentrant=False)
        else:
            text_logits = self.text_head(hidden)

        out: TensorDict = {"text_logits": text_logits}
        if "image" in self.heads:
            if self.use_activation_checkpoint and self.training:
                out["image_pred"] = checkpoint.checkpoint(
                    self.heads["image"], pooled, use_reentrant=False
                )
            else:
                out["image_pred"] = self.heads["image"](pooled)
        if "audio" in self.heads:
            if self.use_activation_checkpoint and self.training:
                out["audio_pred"] = checkpoint.checkpoint(
                    self.heads["audio"], pooled, use_reentrant=False
                )
            else:
                out["audio_pred"] = self.heads["audio"](pooled)
        return out

    def set_tp_group(self, tp_group: Optional[object]) -> None:
        self.text_head.set_tp_group(tp_group)  # type: ignore[arg-type]
        for block in self.blocks:
            block.set_tp_group(tp_group)
        for head in self.heads.values():
            if hasattr(head, "set_tp_group"):
                head.set_tp_group(tp_group)  # type: ignore[misc]
