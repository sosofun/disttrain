from __future__ import annotations

import torch
import torch.nn as nn

from disttrain.models.modalities import register_encoder_modality


@register_encoder_modality("video")
class VideoEncoder(nn.Module):
    def __init__(self, hidden_size: int, image_size: int = 64, frames: int = 8):
        super().__init__()
        input_dim = frames * 3 * image_size * image_size
        self.proj = nn.Linear(input_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        x = video.reshape(video.size(0), -1)
        return self.norm(self.proj(x))
