from __future__ import annotations

import torch
import torch.nn as nn

from disttrain.models.modalities import register_encoder_modality


@register_encoder_modality("audio")
class AudioEncoder(nn.Module):
    def __init__(self, hidden_size: int, audio_length: int = 2048):
        super().__init__()
        input_dim = audio_length
        self.proj = nn.Linear(input_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio expected shape: [B, 1, L] or [B, L]
        if audio.dim() == 3:
            x = audio.squeeze(1)
        else:
            x = audio
        return self.norm(self.proj(x))
