from __future__ import annotations

from typing import Callable, Dict, Type

import torch.nn as nn


_ENCODER_REGISTRY: Dict[str, Type[nn.Module]] = {}
_DECODER_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_encoder_modality(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    def _wrapper(cls: Type[nn.Module]) -> Type[nn.Module]:
        _ENCODER_REGISTRY[name] = cls
        return cls

    return _wrapper


def register_decoder_modality(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    def _wrapper(cls: Type[nn.Module]) -> Type[nn.Module]:
        _DECODER_REGISTRY[name] = cls
        return cls

    return _wrapper


def build_encoder_modality(name: str, **kwargs) -> nn.Module:
    if name not in _ENCODER_REGISTRY:
        raise KeyError(f"unknown encoder modality: {name}")
    return _ENCODER_REGISTRY[name](**kwargs)


def build_decoder_modality(name: str, **kwargs) -> nn.Module:
    if name not in _DECODER_REGISTRY:
        raise KeyError(f"unknown decoder modality: {name}")
    return _DECODER_REGISTRY[name](**kwargs)


# Import built-in modalities to trigger registration.
from disttrain.models.modalities.image_encoder import ImageEncoder  # noqa: E402,F401
from disttrain.models.modalities.video_encoder import VideoEncoder  # noqa: E402,F401
from disttrain.models.modalities.audio_encoder import AudioEncoder  # noqa: E402,F401
from disttrain.models.modalities.image_decoder import ImageDecoderHead  # noqa: E402,F401
from disttrain.models.modalities.audio_decoder import AudioDecoderHead  # noqa: E402,F401
