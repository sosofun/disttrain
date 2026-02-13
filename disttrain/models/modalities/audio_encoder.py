from __future__ import annotations

import torch
import torch.nn as nn

from disttrain.models.modalities import register_encoder_modality
from disttrain.models.tp_layers import ColumnParallelLinear


@register_encoder_modality("audio")
class AudioEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        audio_length: int = 2048,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        super().__init__()
        input_dim = audio_length
        self.proj = ColumnParallelLinear(
            in_features=input_dim,
            out_features=hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            gather_output=True,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def set_tp_group(self, tp_group: object) -> None:
        self.proj.set_tp_group(tp_group)  # type: ignore[arg-type]

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio expected shape: [B, 1, L] or [B, L]
        if audio.dim() == 3:
            x = audio.squeeze(1)
        else:
            x = audio
        return self.norm(self.proj(x))
