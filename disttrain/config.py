from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
import json


STAGE_ORDER = ("encoder", "llm", "decoder")
VALID_MODALITIES = {"text", "image", "video", "audio"}
VALID_SCHEDULES = {"gpipe", "1f1b"}
VALID_TRANSPORT_DTYPES = {"auto", "fp32", "fp16", "bf16"}
VALID_TRANSPORT_TP_MODES = {"single", "auto", "direct"}
VALID_LOSS_WEIGHT_KEYS = {"text", "image", "audio"}
VALID_METRIC_LEVELS = {"off", "minimal", "standard", "detailed", "debug"}
VALID_METRIC_RANK_SCOPES = {"auto", "rank0", "sink", "all"}


class ConfigError(ValueError):
    """Raised when config content is invalid."""


@dataclass
class DistributedConfig:
    backend: str = "nccl"
    world_size: int = 0
    init_method: str = "env://"
    timeout_sec: int = 1800
    grad_sync_bucket_mb: float = 25.0


@dataclass
class StageConfig:
    enabled: bool = False
    tp_size: int = 1
    dp_size: int = 1
    model_cls: str = ""
    activation_checkpoint: bool = False
    sequence_parallel: bool = False
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
    transport_dtype: str = "auto"
    transport_tp_mode: str = "single"


@dataclass
class OptimizerConfig:
    type: str = "adamw"
    lr: float = 2e-4
    weight_decay: float = 0.01
    stage_lrs: Dict[str, float] = field(default_factory=dict)
    zero_stage: int = 0
    zero1_bucket_mb: float = 0.0


@dataclass
class MetricGroupConfig:
    enabled: bool = True
    every_n_steps: int = 1


@dataclass
class MetricsGroupsConfig:
    core: MetricGroupConfig = field(default_factory=MetricGroupConfig)
    throughput: MetricGroupConfig = field(default_factory=MetricGroupConfig)
    comm_summary: MetricGroupConfig = field(default_factory=MetricGroupConfig)
    p2p_detail: MetricGroupConfig = field(default_factory=MetricGroupConfig)
    io: MetricGroupConfig = field(default_factory=MetricGroupConfig)
    memory: MetricGroupConfig = field(default_factory=MetricGroupConfig)


@dataclass
class MetricsConfig:
    level: str = "standard"
    rank_scope: str = "auto"
    log_every_steps: int = 1
    groups: MetricsGroupsConfig = field(
        default_factory=lambda: _default_metrics_groups("standard")
    )


@dataclass
class TrainingConfig:
    global_batch_size: int = 256
    micro_batch_size: int = 4
    grad_accum_steps: int = 1
    grad_clip_norm: float = 1.0
    precision: str = "bf16"
    device: str = "auto"
    deterministic: bool = False
    max_steps: int = 50
    seq_len: int = 128
    vocab_size: int = 32000
    hidden_size: int = 512
    num_attention_heads: int = 8
    image_size: int = 64
    video_frames: int = 8
    audio_length: int = 2048
    data_seed: int = 2026
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {"text": 1.0, "image": 1.0, "audio": 1.0}
    )
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    io: "IOConfig" = field(default_factory=lambda: IOConfig())
    metrics: "MetricsConfig" = field(default_factory=lambda: _default_metrics_config())


@dataclass
class IOConfig:
    enable_prefetch: bool = True
    prefetch_size: int = 2
    pin_memory: bool = True
    num_workers: int = 0


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
            if stage.sequence_parallel and stage.tp_size <= 1:
                raise ConfigError(
                    f"stages.{stage_name}.sequence_parallel requires tp_size > 1, "
                    f"got tp_size={stage.tp_size}"
                )

        if not self.stages["llm"].enabled:
            raise ConfigError("stages.llm.enabled must be true")

        if self.pipeline.schedule not in VALID_SCHEDULES:
            raise ConfigError(
                f"pipeline.schedule must be one of {sorted(VALID_SCHEDULES)}, "
                f"got {self.pipeline.schedule}"
            )
        if self.pipeline.transport_dtype not in VALID_TRANSPORT_DTYPES:
            raise ConfigError(
                "pipeline.transport_dtype must be one of "
                f"{sorted(VALID_TRANSPORT_DTYPES)}, got {self.pipeline.transport_dtype}"
            )
        if self.pipeline.transport_tp_mode not in VALID_TRANSPORT_TP_MODES:
            raise ConfigError(
                "pipeline.transport_tp_mode must be one of "
                f"{sorted(VALID_TRANSPORT_TP_MODES)}, got {self.pipeline.transport_tp_mode}"
            )
        if self.pipeline.num_micro_batches < 1:
            raise ConfigError("pipeline.num_micro_batches must be >= 1")
        pipeline_depth = len(self.enabled_stages)
        if self.pipeline.num_micro_batches < pipeline_depth:
            raise ConfigError(
                "pipeline.num_micro_batches must be >= enabled stage count: "
                f"num_micro_batches={self.pipeline.num_micro_batches}, "
                f"enabled_stage_count={pipeline_depth}"
            )
        if self.pipeline.transport_tp_mode == "direct":
            enabled = self.enabled_stages
            for i in range(len(enabled) - 1):
                lhs = enabled[i]
                rhs = enabled[i + 1]
                lhs_tp = self.stages[lhs].tp_size
                rhs_tp = self.stages[rhs].tp_size
                if lhs_tp != rhs_tp:
                    raise ConfigError(
                        "pipeline.transport_tp_mode=direct requires equal adjacent stage tp_size, "
                        f"got stages.{lhs}.tp_size={lhs_tp}, stages.{rhs}.tp_size={rhs_tp}"
                    )

        if self.distributed.world_size in (0, None):
            self.distributed.world_size = self.expected_world_size
        if self.distributed.world_size != self.expected_world_size:
            raise ConfigError(
                "distributed.world_size mismatch: "
                f"configured={self.distributed.world_size}, "
                f"expected={self.expected_world_size}"
            )
        if self.distributed.grad_sync_bucket_mb < 0:
            raise ConfigError("distributed.grad_sync_bucket_mb must be >= 0")

        if self.training.micro_batch_size < 1:
            raise ConfigError("training.micro_batch_size must be >= 1")
        if self.training.grad_accum_steps < 1:
            raise ConfigError("training.grad_accum_steps must be >= 1")
        if self.training.hidden_size < 1:
            raise ConfigError("training.hidden_size must be >= 1")
        if self.training.num_attention_heads < 1:
            raise ConfigError("training.num_attention_heads must be >= 1")
        if self.training.hidden_size % self.training.num_attention_heads != 0:
            raise ConfigError(
                "training.hidden_size must be divisible by training.num_attention_heads, "
                f"got hidden_size={self.training.hidden_size}, "
                f"num_attention_heads={self.training.num_attention_heads}"
            )
        if self.training.seq_len < 1:
            raise ConfigError("training.seq_len must be >= 1")
        if self.training.data_seed < 0:
            raise ConfigError("training.data_seed must be >= 0")
        if self.training.grad_clip_norm < 0:
            raise ConfigError("training.grad_clip_norm must be >= 0")
        if self.training.device not in {"auto", "cpu", "cuda"}:
            raise ConfigError(
                "training.device must be one of {'auto','cpu','cuda'}, "
                f"got {self.training.device}"
            )
        if self.training.io.prefetch_size < 0:
            raise ConfigError("training.io.prefetch_size must be >= 0")
        if self.training.io.num_workers < 0:
            raise ConfigError("training.io.num_workers must be >= 0")
        if self.training.metrics.level not in VALID_METRIC_LEVELS:
            raise ConfigError(
                "training.metrics.level must be one of "
                f"{sorted(VALID_METRIC_LEVELS)}, got {self.training.metrics.level}"
            )
        if self.training.metrics.rank_scope not in VALID_METRIC_RANK_SCOPES:
            raise ConfigError(
                "training.metrics.rank_scope must be one of "
                f"{sorted(VALID_METRIC_RANK_SCOPES)}, got {self.training.metrics.rank_scope}"
            )
        if self.training.metrics.log_every_steps < 1:
            raise ConfigError("training.metrics.log_every_steps must be >= 1")
        metric_groups = self.training.metrics.groups
        for group_name in (
            "core",
            "throughput",
            "comm_summary",
            "p2p_detail",
            "io",
            "memory",
        ):
            group = getattr(metric_groups, group_name)
            if group.every_n_steps < 1:
                raise ConfigError(
                    f"training.metrics.groups.{group_name}.every_n_steps must be >= 1, "
                    f"got {group.every_n_steps}"
                )
        if not self.training.loss_weights:
            raise ConfigError("training.loss_weights cannot be empty")
        total_loss_weight = 0.0
        for k, v in self.training.loss_weights.items():
            if k not in VALID_LOSS_WEIGHT_KEYS:
                raise ConfigError(
                    "training.loss_weights contains invalid key "
                    f"'{k}', expected one of {sorted(VALID_LOSS_WEIGHT_KEYS)}"
                )
            if v < 0:
                raise ConfigError(
                    f"training.loss_weights.{k} must be >= 0, got {v}"
                )
            total_loss_weight += float(v)
        if total_loss_weight <= 0:
            raise ConfigError("sum(training.loss_weights.values()) must be > 0")

        if self.training.optimizer.type.lower() != "adamw":
            raise ConfigError("training.optimizer.type currently only supports 'adamw'")
        if self.training.optimizer.zero_stage not in {0, 1}:
            raise ConfigError(
                "training.optimizer.zero_stage must be 0 or 1, "
                f"got {self.training.optimizer.zero_stage}"
            )
        if self.training.optimizer.zero1_bucket_mb < 0:
            raise ConfigError("training.optimizer.zero1_bucket_mb must be >= 0")
        for stage_name in STAGE_ORDER:
            stage = self.stages[stage_name]
            if not stage.enabled:
                continue
            if self.training.num_attention_heads % stage.tp_size != 0:
                raise ConfigError(
                    "training.num_attention_heads must be divisible by stage tp_size "
                    f"for enabled stage '{stage_name}', got "
                    f"num_attention_heads={self.training.num_attention_heads}, "
                    f"tp_size={stage.tp_size}"
                )
        for k, v in self.training.optimizer.stage_lrs.items():
            if k not in STAGE_ORDER:
                raise ConfigError(
                    "training.optimizer.stage_lrs contains invalid stage key "
                    f"'{k}', expected one of {list(STAGE_ORDER)}"
                )
            if v <= 0:
                raise ConfigError(
                    "training.optimizer.stage_lrs values must be > 0, "
                    f"got {k}={v}"
                )

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
            grad_sync_bucket_mb=float(distributed_raw.get("grad_sync_bucket_mb", 25.0)),
        )
        pipeline = PipelineConfig(
            schedule=str(pipeline_raw.get("schedule", "1f1b")).lower(),
            num_micro_batches=int(pipeline_raw.get("num_micro_batches", 8)),
            overlap_p2p_comm=bool(pipeline_raw.get("overlap_p2p_comm", True)),
            transport_dtype=str(pipeline_raw.get("transport_dtype", "auto")).lower(),
            transport_tp_mode=str(pipeline_raw.get("transport_tp_mode", "single")).lower(),
        )
        optimizer_raw = training_raw.get("optimizer", {})
        io_raw = training_raw.get("io", {})
        metrics_raw = training_raw.get("metrics", {})
        if metrics_raw is None:
            metrics_raw = {}
        loss_weights_raw = training_raw.get("loss_weights", {})
        if loss_weights_raw is None:
            loss_weights_raw = {}
        # Start from defaults so users can override only a subset of tasks.
        loss_weights = {"text": 1.0, "image": 1.0, "audio": 1.0}
        for k, v in dict(loss_weights_raw).items():
            loss_weights[str(k)] = float(v)
        optimizer = OptimizerConfig(
            type=str(optimizer_raw.get("type", "adamw")),
            lr=float(optimizer_raw.get("lr", 2e-4)),
            weight_decay=float(optimizer_raw.get("weight_decay", 0.01)),
            stage_lrs={
                str(k): float(v)
                for k, v in (optimizer_raw.get("stage_lrs", {}) or {}).items()
            },
            zero_stage=int(optimizer_raw.get("zero_stage", 0)),
            zero1_bucket_mb=float(optimizer_raw.get("zero1_bucket_mb", 0.0)),
        )
        io_cfg = IOConfig(
            enable_prefetch=bool(io_raw.get("enable_prefetch", True)),
            prefetch_size=int(io_raw.get("prefetch_size", 2)),
            pin_memory=bool(io_raw.get("pin_memory", True)),
            num_workers=int(io_raw.get("num_workers", 0)),
        )
        metrics_level = str(metrics_raw.get("level", "standard")).lower()
        default_metrics_groups = _default_metrics_groups(metrics_level)
        groups_raw = metrics_raw.get("groups", {})
        if not isinstance(groups_raw, Mapping):
            groups_raw = {}
        metrics_groups = MetricsGroupsConfig(
            core=_merge_metric_group(groups_raw.get("core", {}), default_metrics_groups.core),
            throughput=_merge_metric_group(
                groups_raw.get("throughput", {}),
                default_metrics_groups.throughput,
            ),
            comm_summary=_merge_metric_group(
                groups_raw.get("comm_summary", {}),
                default_metrics_groups.comm_summary,
            ),
            p2p_detail=_merge_metric_group(
                groups_raw.get("p2p_detail", {}),
                default_metrics_groups.p2p_detail,
            ),
            io=_merge_metric_group(groups_raw.get("io", {}), default_metrics_groups.io),
            memory=_merge_metric_group(
                groups_raw.get("memory", {}),
                default_metrics_groups.memory,
            ),
        )
        metrics_cfg = MetricsConfig(
            level=metrics_level,
            rank_scope=str(metrics_raw.get("rank_scope", "auto")).lower(),
            log_every_steps=int(metrics_raw.get("log_every_steps", 1)),
            groups=metrics_groups,
        )
        training = TrainingConfig(
            global_batch_size=int(training_raw.get("global_batch_size", 256)),
            micro_batch_size=int(training_raw.get("micro_batch_size", 4)),
            grad_accum_steps=int(training_raw.get("grad_accum_steps", 1)),
            grad_clip_norm=float(training_raw.get("grad_clip_norm", 1.0)),
            precision=str(training_raw.get("precision", "bf16")).lower(),
            device=str(training_raw.get("device", "auto")).lower(),
            deterministic=bool(training_raw.get("deterministic", False)),
            max_steps=int(training_raw.get("max_steps", 50)),
            seq_len=int(training_raw.get("seq_len", 128)),
            vocab_size=int(training_raw.get("vocab_size", 32000)),
            hidden_size=int(training_raw.get("hidden_size", 512)),
            num_attention_heads=int(training_raw.get("num_attention_heads", 8)),
            image_size=int(training_raw.get("image_size", 64)),
            video_frames=int(training_raw.get("video_frames", 8)),
            audio_length=int(training_raw.get("audio_length", 2048)),
            data_seed=int(training_raw.get("data_seed", 2026)),
            loss_weights=loss_weights,
            optimizer=optimizer,
            io=io_cfg,
            metrics=metrics_cfg,
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
                activation_checkpoint=bool(stage_raw.get("activation_checkpoint", False)),
                sequence_parallel=bool(stage_raw.get("sequence_parallel", False)),
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


def _default_metrics_config() -> MetricsConfig:
    return MetricsConfig()


def _default_metrics_groups(level: str) -> MetricsGroupsConfig:
    normalized = level.lower()
    if normalized in {"detailed", "debug"}:
        return MetricsGroupsConfig(
            core=MetricGroupConfig(enabled=True, every_n_steps=1),
            throughput=MetricGroupConfig(enabled=True, every_n_steps=1),
            comm_summary=MetricGroupConfig(enabled=True, every_n_steps=1),
            p2p_detail=MetricGroupConfig(enabled=True, every_n_steps=1),
            io=MetricGroupConfig(enabled=True, every_n_steps=1),
            memory=MetricGroupConfig(enabled=True, every_n_steps=1),
        )
    if normalized == "minimal":
        return MetricsGroupsConfig(
            core=MetricGroupConfig(enabled=True, every_n_steps=1),
            throughput=MetricGroupConfig(enabled=True, every_n_steps=1),
            comm_summary=MetricGroupConfig(enabled=False, every_n_steps=1),
            p2p_detail=MetricGroupConfig(enabled=False, every_n_steps=1),
            io=MetricGroupConfig(enabled=False, every_n_steps=1),
            memory=MetricGroupConfig(enabled=False, every_n_steps=1),
        )
    if normalized == "off":
        return MetricsGroupsConfig(
            core=MetricGroupConfig(enabled=False, every_n_steps=1),
            throughput=MetricGroupConfig(enabled=False, every_n_steps=1),
            comm_summary=MetricGroupConfig(enabled=False, every_n_steps=1),
            p2p_detail=MetricGroupConfig(enabled=False, every_n_steps=1),
            io=MetricGroupConfig(enabled=False, every_n_steps=1),
            memory=MetricGroupConfig(enabled=False, every_n_steps=1),
        )
    # standard (default): keep throughput + summary + I/O, disable costly p2p/memory.
    return MetricsGroupsConfig(
        core=MetricGroupConfig(enabled=True, every_n_steps=1),
        throughput=MetricGroupConfig(enabled=True, every_n_steps=1),
        comm_summary=MetricGroupConfig(enabled=True, every_n_steps=1),
        p2p_detail=MetricGroupConfig(enabled=False, every_n_steps=1),
        io=MetricGroupConfig(enabled=True, every_n_steps=1),
        memory=MetricGroupConfig(enabled=False, every_n_steps=1),
    )


def _merge_metric_group(raw: Any, default: MetricGroupConfig) -> MetricGroupConfig:
    if not isinstance(raw, Mapping):
        raw = {}
    return MetricGroupConfig(
        enabled=bool(raw.get("enabled", default.enabled)),
        every_n_steps=int(raw.get("every_n_steps", default.every_n_steps)),
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
