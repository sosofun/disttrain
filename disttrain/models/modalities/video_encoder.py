from __future__ import annotations

import torch
import torch.nn as nn

from disttrain.models.modalities import register_encoder_modality
from disttrain.models.tp_layers import ColumnParallelLinear


@register_encoder_modality("video")
class VideoEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        image_size: int = 64,
        frames: int = 8,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        super().__init__()
        input_dim = frames * 3 * image_size * image_size
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

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        x = video.reshape(video.size(0), -1)
        return self.norm(self.proj(x))
