from __future__ import annotations

import torch
import torch.nn as nn

from disttrain.models.modalities import register_decoder_modality


@register_decoder_modality("audio")
class AudioDecoderHead(nn.Module):
    def __init__(self, hidden_size: int, audio_length: int = 2048):
        super().__init__()
        self.proj = nn.Linear(hidden_size, audio_length)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = self.proj(hidden)
        return x.unsqueeze(1)
