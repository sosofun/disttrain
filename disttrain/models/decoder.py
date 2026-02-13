from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from disttrain.config import StageConfig, TrainingConfig
from disttrain.models.base import StageModel, TensorDict
from disttrain.models.modalities import build_decoder_modality


class DecoderModel(StageModel):
    stage_name = "decoder"

    def __init__(self, stage_cfg: StageConfig, train_cfg: TrainingConfig):
        super().__init__()
        self.output_modalities = list(stage_cfg.output_modalities)
        self.use_activation_checkpoint = stage_cfg.activation_checkpoint
        self.text_head = nn.Linear(train_cfg.hidden_size, train_cfg.vocab_size)

        heads: Dict[str, nn.Module] = {}
        if "image" in self.output_modalities:
            heads["image"] = build_decoder_modality(
                "image",
                hidden_size=train_cfg.hidden_size,
                image_size=train_cfg.image_size,
            )
        if "audio" in self.output_modalities:
            heads["audio"] = build_decoder_modality(
                "audio",
                hidden_size=train_cfg.hidden_size,
                audio_length=train_cfg.audio_length,
            )
        self.heads = nn.ModuleDict(heads)

    def forward(
        self, inputs: TensorDict, meta: Optional[Dict[str, object]] = None
    ) -> TensorDict:
        if "hidden_states" not in inputs:
            raise KeyError("DecoderModel expects 'hidden_states'")
        hidden = inputs["hidden_states"]
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
