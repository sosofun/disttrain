from __future__ import annotations

from typing import Dict, Type

from disttrain.config import StageConfig, TrainingConfig
from disttrain.models.base import StageModel
from disttrain.models.encoder import EncoderModel
from disttrain.models.llm import LLMModel
from disttrain.models.decoder import DecoderModel


_STAGE_MODEL_REGISTRY: Dict[str, Type[StageModel]] = {
    "EncoderModel": EncoderModel,
    "LLMModel": LLMModel,
    "DecoderModel": DecoderModel,
}


def register_stage_model(name: str, cls: Type[StageModel]) -> None:
    _STAGE_MODEL_REGISTRY[name] = cls


def build_stage_model(stage_cfg: StageConfig, train_cfg: TrainingConfig) -> StageModel:
    if stage_cfg.model_cls not in _STAGE_MODEL_REGISTRY:
        raise KeyError(
            f"unknown stage model_cls '{stage_cfg.model_cls}', "
            f"registered={sorted(_STAGE_MODEL_REGISTRY)}"
        )
    model_cls = _STAGE_MODEL_REGISTRY[stage_cfg.model_cls]
    return model_cls(stage_cfg, train_cfg)
