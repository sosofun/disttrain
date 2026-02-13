from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
import json


STAGE_ORDER = ("encoder", "llm", "decoder")
VALID_MODALITIES = {"text", "image", "video", "audio"}
VALID_SCHEDULES = {"gpipe", "1f1b"}


class ConfigError(ValueError):
    """Raised when config content is invalid."""


@dataclass
class DistributedConfig:
    backend: str = "nccl"
    world_size: int = 0
    init_method: str = "env://"
    timeout_sec: int = 1800


@dataclass
class StageConfig:
    enabled: bool = False
    tp_size: int = 1
    dp_size: int = 1
    model_cls: str = ""
    input_modalities: List[str] = field(default_factory=list)
    output_modalities: List[str] = field(default_factory=list)

    @property
    def world_size(self) -> int:
        if not self.enabled:
            return 0
        return self.tp_size * self.dp_size


@dataclass
class PipelineConfig:
    schedule: str = "1f1b"
    num_micro_batches: int = 8
    overlap_p2p_comm: bool = True


@dataclass
class OptimizerConfig:
    type: str = "adamw"
    lr: float = 2e-4
    weight_decay: float = 0.01


@dataclass
class TrainingConfig:
    global_batch_size: int = 256
    micro_batch_size: int = 4
    grad_accum_steps: int = 1
    precision: str = "bf16"
    device: str = "auto"
    max_steps: int = 50
    seq_len: int = 128
    vocab_size: int = 32000
    hidden_size: int = 512
    image_size: int = 64
    video_frames: int = 8
    audio_length: int = 2048
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass
class RunConfig:
    distributed: DistributedConfig
    stages: Dict[str, StageConfig]
    pipeline: PipelineConfig
    training: TrainingConfig

    @property
    def enabled_stages(self) -> List[str]:
        return [name for name in STAGE_ORDER if self.stages[name].enabled]

    @property
    def expected_world_size(self) -> int:
        return sum(self.stages[name].world_size for name in STAGE_ORDER)

    def validate(self) -> None:
        for stage_name in STAGE_ORDER:
            stage = self.stages[stage_name]
            if not stage.enabled:
                continue
            if stage.tp_size < 1 or stage.dp_size < 1:
                raise ConfigError(
                    f"stages.{stage_name}: tp_size and dp_size must be >= 1, "
                    f"got tp={stage.tp_size}, dp={stage.dp_size}"
                )
            if not stage.model_cls:
                raise ConfigError(f"stages.{stage_name}.model_cls cannot be empty")
            _validate_modalities(stage_name, stage.input_modalities, "input_modalities")
            _validate_modalities(stage_name, stage.output_modalities, "output_modalities")

        if not self.stages["llm"].enabled:
            raise ConfigError("stages.llm.enabled must be true")

        if self.pipeline.schedule not in VALID_SCHEDULES:
            raise ConfigError(
                f"pipeline.schedule must be one of {sorted(VALID_SCHEDULES)}, "
                f"got {self.pipeline.schedule}"
            )
        if self.pipeline.num_micro_batches < 1:
            raise ConfigError("pipeline.num_micro_batches must be >= 1")

        if self.distributed.world_size in (0, None):
            self.distributed.world_size = self.expected_world_size
        if self.distributed.world_size != self.expected_world_size:
            raise ConfigError(
                "distributed.world_size mismatch: "
                f"configured={self.distributed.world_size}, "
                f"expected={self.expected_world_size}"
            )

        if self.training.micro_batch_size < 1:
            raise ConfigError("training.micro_batch_size must be >= 1")
        if self.training.grad_accum_steps < 1:
            raise ConfigError("training.grad_accum_steps must be >= 1")
        if self.training.hidden_size < 1:
            raise ConfigError("training.hidden_size must be >= 1")
        if self.training.seq_len < 1:
            raise ConfigError("training.seq_len must be >= 1")
        if self.training.device not in {"auto", "cpu", "cuda"}:
            raise ConfigError(
                "training.device must be one of {'auto','cpu','cuda'}, "
                f"got {self.training.device}"
            )

        if self.training.optimizer.type.lower() != "adamw":
            raise ConfigError("training.optimizer.type currently only supports 'adamw'")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RunConfig":
        distributed_raw = raw.get("distributed", {})
        pipeline_raw = raw.get("pipeline", {})
        training_raw = raw.get("training", {})
        stages_raw = raw.get("stages", {})

        distributed = DistributedConfig(
            backend=str(distributed_raw.get("backend", "nccl")).lower(),
            world_size=int(distributed_raw.get("world_size", 0) or 0),
            init_method=str(distributed_raw.get("init_method", "env://")),
            timeout_sec=int(distributed_raw.get("timeout_sec", 1800)),
        )
        pipeline = PipelineConfig(
            schedule=str(pipeline_raw.get("schedule", "1f1b")).lower(),
            num_micro_batches=int(pipeline_raw.get("num_micro_batches", 8)),
            overlap_p2p_comm=bool(pipeline_raw.get("overlap_p2p_comm", True)),
        )
        optimizer_raw = training_raw.get("optimizer", {})
        optimizer = OptimizerConfig(
            type=str(optimizer_raw.get("type", "adamw")),
            lr=float(optimizer_raw.get("lr", 2e-4)),
            weight_decay=float(optimizer_raw.get("weight_decay", 0.01)),
        )
        training = TrainingConfig(
            global_batch_size=int(training_raw.get("global_batch_size", 256)),
            micro_batch_size=int(training_raw.get("micro_batch_size", 4)),
            grad_accum_steps=int(training_raw.get("grad_accum_steps", 1)),
            precision=str(training_raw.get("precision", "bf16")).lower(),
            device=str(training_raw.get("device", "auto")).lower(),
            max_steps=int(training_raw.get("max_steps", 50)),
            seq_len=int(training_raw.get("seq_len", 128)),
            vocab_size=int(training_raw.get("vocab_size", 32000)),
            hidden_size=int(training_raw.get("hidden_size", 512)),
            image_size=int(training_raw.get("image_size", 64)),
            video_frames=int(training_raw.get("video_frames", 8)),
            audio_length=int(training_raw.get("audio_length", 2048)),
            optimizer=optimizer,
        )

        stages: Dict[str, StageConfig] = {}
        for stage_name in STAGE_ORDER:
            default_enabled = stage_name == "llm"
            stage_raw = stages_raw.get(stage_name, {})
            stages[stage_name] = StageConfig(
                enabled=bool(stage_raw.get("enabled", default_enabled)),
                tp_size=int(stage_raw.get("tp_size", 1)),
                dp_size=int(stage_raw.get("dp_size", 1)),
                model_cls=str(
                    stage_raw.get(
                        "model_cls",
                        {
                            "encoder": "EncoderModel",
                            "llm": "LLMModel",
                            "decoder": "DecoderModel",
                        }[stage_name],
                    )
                ),
                input_modalities=[str(x) for x in stage_raw.get("input_modalities", [])],
                output_modalities=[str(x) for x in stage_raw.get("output_modalities", [])],
            )

        cfg = cls(
            distributed=distributed,
            stages=stages,
            pipeline=pipeline,
            training=training,
        )
        cfg.validate()
        return cfg


def _validate_modalities(stage_name: str, values: List[str], field_name: str) -> None:
    for value in values:
        if value not in VALID_MODALITIES:
            raise ConfigError(
                f"stages.{stage_name}.{field_name} contains invalid modality '{value}', "
                f"expected one of {sorted(VALID_MODALITIES)}"
            )


def load_config(path: str) -> RunConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"config file not found: {path}")

    suffix = config_path.suffix.lower()
    if suffix == ".json":
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ConfigError(
                "YAML config requires PyYAML. Install with `pip install pyyaml`."
            ) from exc
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    else:
        raise ConfigError(f"unsupported config suffix: {suffix}")

    if not isinstance(raw, Mapping):
        raise ConfigError("top-level config must be a mapping/object")
    return RunConfig.from_dict(raw)
