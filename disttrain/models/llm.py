from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from disttrain.config import StageConfig, TrainingConfig
from disttrain.models.base import StageModel, TensorDict
from disttrain.models.tp_layers import ColumnParallelLinear, VocabParallelEmbedding
from disttrain.models.tp_transformer import TPTransformerBlock


class LLMModel(StageModel):
    stage_name = "llm"

    def __init__(
        self,
        stage_cfg: StageConfig,
        train_cfg: TrainingConfig,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        super().__init__()
        self.hidden_size = train_cfg.hidden_size
        self.vocab_size = train_cfg.vocab_size
        self.use_activation_checkpoint = stage_cfg.activation_checkpoint
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.token_embedding = VocabParallelEmbedding(
            train_cfg.vocab_size,
            train_cfg.hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        self.norm = nn.LayerNorm(train_cfg.hidden_size)

        self.layers = nn.ModuleList(
            [
                TPTransformerBlock(
                    hidden_size=train_cfg.hidden_size,
                    num_heads=train_cfg.num_attention_heads,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    causal=True,
                )
                for _ in range(2)
            ]
        )
        self.lm_head = ColumnParallelLinear(
            train_cfg.hidden_size,
            train_cfg.vocab_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            gather_output=True,
        )

    def set_tp_group(self, tp_group: Optional[object]) -> None:
        self.token_embedding.set_tp_group(tp_group)  # type: ignore[arg-type]
        self.lm_head.set_tp_group(tp_group)  # type: ignore[arg-type]
        for layer in self.layers:
            if hasattr(layer, "set_tp_group"):
                layer.set_tp_group(tp_group)  # type: ignore[misc]

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
            if self.use_activation_checkpoint and self.training:
                hidden = checkpoint.checkpoint(layer, hidden, use_reentrant=False)
            else:
                hidden = layer(hidden)

        logits = self.lm_head(self.norm(hidden))
        return {"hidden_states": hidden, "logits": logits}
