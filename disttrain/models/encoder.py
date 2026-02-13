from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from disttrain.config import StageConfig, TrainingConfig
from disttrain.models.base import StageModel, TensorDict
from disttrain.models.modalities import build_encoder_modality
from disttrain.models.tp_layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)


class TPFusionBlock(nn.Module):
    def __init__(self, hidden_size: int, tp_size: int, tp_rank: int):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            gather_output=False,
        )
        self.act = nn.GELU()
        self.fc2 = RowParallelLinear(
            hidden_size,
            hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            input_is_parallel=True,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def set_tp_group(self, tp_group: Optional[object]) -> None:
        self.fc1.set_tp_group(tp_group)  # type: ignore[arg-type]
        self.fc2.set_tp_group(tp_group)  # type: ignore[arg-type]

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(hidden)
        hidden = self.act(hidden)
        hidden = self.fc2(hidden)
        hidden = self.norm(hidden)
        return hidden


class EncoderModel(StageModel):
    stage_name = "encoder"

    def __init__(
        self,
        stage_cfg: StageConfig,
        train_cfg: TrainingConfig,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        super().__init__()
        self.hidden_size = train_cfg.hidden_size
        self.input_modalities = list(stage_cfg.input_modalities)
        self.use_activation_checkpoint = stage_cfg.activation_checkpoint
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.text_embedding = VocabParallelEmbedding(
            train_cfg.vocab_size,
            train_cfg.hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        self.fusion = TPFusionBlock(
            hidden_size=train_cfg.hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
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

    def set_tp_group(self, tp_group: Optional[object]) -> None:
        self.text_embedding.set_tp_group(tp_group)  # type: ignore[arg-type]
        self.fusion.set_tp_group(tp_group)  # type: ignore[arg-type]

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
        if self.use_activation_checkpoint and self.training:
            hidden = checkpoint.checkpoint(self.fusion, hidden, use_reentrant=False)
        else:
            hidden = self.fusion(hidden)
        return {"hidden_states": hidden}
