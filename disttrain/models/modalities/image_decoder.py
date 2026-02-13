from __future__ import annotations

import torch
import torch.nn as nn

from disttrain.models.modalities import register_decoder_modality


@register_decoder_modality("image")
class ImageDecoderHead(nn.Module):
    def __init__(self, hidden_size: int, image_size: int = 64):
        super().__init__()
        output_dim = 3 * image_size * image_size
        self.proj = nn.Linear(hidden_size, output_dim)
        self.image_size = image_size

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = self.proj(hidden)
        return x.reshape(hidden.size(0), 3, self.image_size, self.image_size)
