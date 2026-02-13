from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from disttrain.config import StageConfig, TrainingConfig
from disttrain.models.base import StageModel, TensorDict
from disttrain.models.modalities import build_encoder_modality


class EncoderModel(StageModel):
    stage_name = "encoder"

    def __init__(self, stage_cfg: StageConfig, train_cfg: TrainingConfig):
        super().__init__()
        self.hidden_size = train_cfg.hidden_size
        self.input_modalities = list(stage_cfg.input_modalities)
        self.text_embedding = nn.Embedding(train_cfg.vocab_size, train_cfg.hidden_size)
        self.fusion = nn.Sequential(
            nn.Linear(train_cfg.hidden_size, train_cfg.hidden_size),
            nn.GELU(),
            nn.LayerNorm(train_cfg.hidden_size),
        )

        branches: Dict[str, nn.Module] = {}
        if "image" in self.input_modalities:
            branches["image"] = build_encoder_modality(
                "image",
                hidden_size=train_cfg.hidden_size,
                image_size=train_cfg.image_size,
            )
        if "video" in self.input_modalities:
            branches["video"] = build_encoder_modality(
                "video",
                hidden_size=train_cfg.hidden_size,
                image_size=train_cfg.image_size,
                frames=train_cfg.video_frames,
            )
        if "audio" in self.input_modalities:
            branches["audio"] = build_encoder_modality(
                "audio",
                hidden_size=train_cfg.hidden_size,
                audio_length=train_cfg.audio_length,
            )
        self.branches = nn.ModuleDict(branches)

    def forward(
        self, inputs: TensorDict, meta: Optional[Dict[str, object]] = None
    ) -> TensorDict:
        if "text_tokens" not in inputs:
            raise KeyError("EncoderModel expects 'text_tokens' in inputs")

        text_tokens = inputs["text_tokens"]
        hidden = self.text_embedding(text_tokens)
        fused_bias = torch.zeros(
            hidden.size(0), self.hidden_size, dtype=hidden.dtype, device=hidden.device
        )

        for name, branch in self.branches.items():
            if name not in inputs:
                continue
            fused_bias = fused_bias + branch(inputs[name])

        hidden = hidden + fused_bias.unsqueeze(1)
        hidden = self.fusion(hidden)
        return {"hidden_states": hidden}
