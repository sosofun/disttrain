from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

from disttrain.config import RunConfig, STAGE_ORDER


@dataclass(frozen=True)
class StagePlacement:
    name: str
    enabled: bool
    start_rank: int
    end_rank: int
    tp_size: int
    dp_size: int
    model_cls: str
    input_modalities: List[str]
    output_modalities: List[str]

    @property
    def world_size(self) -> int:
        if not self.enabled:
            return 0
        return self.end_rank - self.start_rank + 1

    def contains(self, rank: int) -> bool:
        if not self.enabled:
            return False
        return self.start_rank <= rank <= self.end_rank


class TopologyError(ValueError):
    """Raised when runtime topology is invalid."""


class Topology:
    def __init__(self, config: RunConfig, runtime_world_size: int, runtime_rank: int):
        self.config = config
        self.runtime_world_size = runtime_world_size
        self.runtime_rank = runtime_rank
        self.stages: Dict[str, StagePlacement] = {}

        cursor = 0
        for stage_name in STAGE_ORDER:
            stage_cfg = config.stages[stage_name]
            if not stage_cfg.enabled:
                placement = StagePlacement(
                    name=stage_name,
                    enabled=False,
                    start_rank=cursor,
                    end_rank=cursor - 1,
                    tp_size=stage_cfg.tp_size,
                    dp_size=stage_cfg.dp_size,
                    model_cls=stage_cfg.model_cls,
                    input_modalities=list(stage_cfg.input_modalities),
                    output_modalities=list(stage_cfg.output_modalities),
                )
                self.stages[stage_name] = placement
                continue

            stage_world = stage_cfg.tp_size * stage_cfg.dp_size
            start_rank = cursor
            end_rank = cursor + stage_world - 1
            cursor += stage_world
            placement = StagePlacement(
                name=stage_name,
                enabled=True,
                start_rank=start_rank,
                end_rank=end_rank,
                tp_size=stage_cfg.tp_size,
                dp_size=stage_cfg.dp_size,
                model_cls=stage_cfg.model_cls,
                input_modalities=list(stage_cfg.input_modalities),
                output_modalities=list(stage_cfg.output_modalities),
            )
            self.stages[stage_name] = placement

        expected_world = cursor
        if expected_world != runtime_world_size:
            raise TopologyError(
                "runtime world size mismatch: "
                f"runtime={runtime_world_size}, expected_by_config={expected_world}"
            )
        if runtime_rank < 0 or runtime_rank >= runtime_world_size:
            raise TopologyError(
                f"runtime rank out of range: rank={runtime_rank}, world={runtime_world_size}"
            )

        local = self.stage_of_rank(runtime_rank)
        if local is None:
            raise TopologyError(f"rank {runtime_rank} is not assigned to any enabled stage")
        self._local_stage_name = local.name

    @property
    def enabled_stage_names(self) -> List[str]:
        return [name for name in STAGE_ORDER if self.stages[name].enabled]

    @property
    def local_stage(self) -> StagePlacement:
        return self.stages[self._local_stage_name]

    @property
    def local_stage_name(self) -> str:
        return self._local_stage_name

    @property
    def pipeline_depth(self) -> int:
        return len(self.enabled_stage_names)

    def stage_of_rank(self, rank: int) -> Optional[StagePlacement]:
        for stage_name in STAGE_ORDER:
            placement = self.stages[stage_name]
            if placement.contains(rank):
                return placement
        return None

    def stage_rank(self, rank: int) -> int:
        stage = self.stage_of_rank(rank)
        if stage is None:
            raise TopologyError(f"rank {rank} has no stage placement")
        return rank - stage.start_rank

    def local_stage_rank(self) -> int:
        return self.stage_rank(self.runtime_rank)

    def local_tp_index(self) -> int:
        stage_rank = self.local_stage_rank()
        return stage_rank % self.local_stage.tp_size

    def local_dp_index(self) -> int:
        stage_rank = self.local_stage_rank()
        return stage_rank // self.local_stage.tp_size

    def rank_for(self, stage_name: str, dp_idx: int, tp_idx: int) -> int:
        stage = self.stages[stage_name]
        if not stage.enabled:
            raise TopologyError(f"stage {stage_name} is disabled")
        if dp_idx < 0 or dp_idx >= stage.dp_size:
            raise TopologyError(
                f"dp index out of range for stage {stage_name}: {dp_idx}"
            )
        if tp_idx < 0 or tp_idx >= stage.tp_size:
            raise TopologyError(
                f"tp index out of range for stage {stage_name}: {tp_idx}"
            )
        return stage.start_rank + dp_idx * stage.tp_size + tp_idx

    def next_stage(self, stage_name: str) -> Optional[str]:
        enabled = self.enabled_stage_names
        idx = enabled.index(stage_name)
        if idx + 1 >= len(enabled):
            return None
        return enabled[idx + 1]

    def prev_stage(self, stage_name: str) -> Optional[str]:
        enabled = self.enabled_stage_names
        idx = enabled.index(stage_name)
        if idx - 1 < 0:
            return None
        return enabled[idx - 1]

    def vps_size_for_boundary(self, from_stage: str, to_stage: str) -> int:
        from_dp = self.stages[from_stage].dp_size
        to_dp = self.stages[to_stage].dp_size
        return math.lcm(from_dp, to_dp)

    def boundary_dp_mapping(
        self, from_stage: str, to_stage: str
    ) -> List[Tuple[int, int, int]]:
        """
        Returns a list of (vps_id, from_dp_replica, to_dp_replica).
        """
        vps_size = self.vps_size_for_boundary(from_stage, to_stage)
        from_dp = self.stages[from_stage].dp_size
        to_dp = self.stages[to_stage].dp_size
        mapping: List[Tuple[int, int, int]] = []
        for vps in range(vps_size):
            mapping.append((vps, vps % from_dp, vps % to_dp))
        return mapping

    def summary_lines(self) -> List[str]:
        lines = [
            "==== Topology Summary ====",
            f"runtime_rank={self.runtime_rank}/{self.runtime_world_size}",
            f"enabled_stages={self.enabled_stage_names}",
        ]
        for stage_name in STAGE_ORDER:
            stage = self.stages[stage_name]
            if not stage.enabled:
                lines.append(f"- {stage_name}: disabled")
                continue
            lines.append(
                f"- {stage_name}: ranks=[{stage.start_rank},{stage.end_rank}], "
                f"tp={stage.tp_size}, dp={stage.dp_size}, model_cls={stage.model_cls}, "
                f"input_modalities={stage.input_modalities}, "
                f"output_modalities={stage.output_modalities}"
            )
        lines.append(
            f"local_stage={self.local_stage_name}, "
            f"local_tp={self.local_tp_index()}, local_dp={self.local_dp_index()}"
        )
        return lines
