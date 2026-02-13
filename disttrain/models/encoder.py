from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from disttrain.config import StageConfig, TrainingConfig
from disttrain.models.base import StageModel, TensorDict
from disttrain.models.modalities import build_encoder_modality
from disttrain.models.tp_layers import VocabParallelEmbedding
from disttrain.models.tp_transformer import TPTransformerBlock


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

        branches: Dict[str, nn.Module] = {}
        if "image" in self.input_modalities:
            branches["image"] = build_encoder_modality(
                "image",
                hidden_size=train_cfg.hidden_size,
                image_size=train_cfg.image_size,
                tp_size=tp_size,
                tp_rank=tp_rank,
            )
        if "video" in self.input_modalities:
            branches["video"] = build_encoder_modality(
                "video",
                hidden_size=train_cfg.hidden_size,
                image_size=train_cfg.image_size,
                frames=train_cfg.video_frames,
                tp_size=tp_size,
                tp_rank=tp_rank,
            )
        if "audio" in self.input_modalities:
            branches["audio"] = build_encoder_modality(
                "audio",
                hidden_size=train_cfg.hidden_size,
                audio_length=train_cfg.audio_length,
                tp_size=tp_size,
                tp_rank=tp_rank,
            )
        self.branches = nn.ModuleDict(branches)

    def set_tp_group(self, tp_group: Optional[object]) -> None:
        self.text_embedding.set_tp_group(tp_group)  # type: ignore[arg-type]
        for block in self.blocks:
            block.set_tp_group(tp_group)
        for branch in self.branches.values():
            if hasattr(branch, "set_tp_group"):
                branch.set_tp_group(tp_group)  # type: ignore[misc]

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
        for block in self.blocks:
            if self.use_activation_checkpoint and self.training:
                hidden = checkpoint.checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        hidden = self.norm(hidden)
        return {"hidden_states": hidden}
