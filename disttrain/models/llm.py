from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from disttrain.config import StageConfig, TrainingConfig
from disttrain.models.base import StageModel, TensorDict


class LLMModel(StageModel):
    stage_name = "llm"

    def __init__(self, stage_cfg: StageConfig, train_cfg: TrainingConfig):
        super().__init__()
        self.hidden_size = train_cfg.hidden_size
        self.vocab_size = train_cfg.vocab_size
        self.token_embedding = nn.Embedding(train_cfg.vocab_size, train_cfg.hidden_size)

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(train_cfg.hidden_size, train_cfg.hidden_size),
                    nn.GELU(),
                    nn.LayerNorm(train_cfg.hidden_size),
                )
                for _ in range(2)
            ]
        )
        self.lm_head = nn.Linear(train_cfg.hidden_size, train_cfg.vocab_size)

    def forward(
        self, inputs: TensorDict, meta: Optional[Dict[str, object]] = None
    ) -> TensorDict:
        if "hidden_states" in inputs:
            hidden = inputs["hidden_states"]
        elif "text_tokens" in inputs:
            hidden = self.token_embedding(inputs["text_tokens"])
        else:
            raise KeyError("LLMModel expects either 'hidden_states' or 'text_tokens'")

        for layer in self.layers:
            hidden = layer(hidden)

        logits = self.lm_head(hidden)
        return {"hidden_states": hidden, "logits": logits}
