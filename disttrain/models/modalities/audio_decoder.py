from __future__ import annotations

import torch
import torch.nn as nn

from disttrain.models.modalities import register_decoder_modality
from disttrain.models.tp_layers import RowParallelLinear


@register_decoder_modality("audio")
class AudioDecoderHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        audio_length: int = 2048,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        super().__init__()
        self.proj = RowParallelLinear(
            in_features=hidden_size,
            out_features=audio_length,
            tp_size=tp_size,
            tp_rank=tp_rank,
            input_is_parallel=False,
        )

    def set_tp_group(self, tp_group: object) -> None:
        self.proj.set_tp_group(tp_group)  # type: ignore[arg-type]

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = self.proj(hidden)
        return x.unsqueeze(1)
