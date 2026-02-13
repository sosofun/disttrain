from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ScheduleAction:
    kind: str  # "F" or "B"
    micro_batch_idx: int


class PipelineScheduler:
    def __init__(
        self,
        schedule: str,
        num_micro_batches: int,
        pipeline_depth: int,
        stage_index: int,
    ):
        self.schedule = schedule.lower()
        self.num_micro_batches = num_micro_batches
        self.pipeline_depth = pipeline_depth
        self.stage_index = stage_index

    def build(self) -> List[ScheduleAction]:
        if self.schedule == "gpipe":
            return self._build_gpipe()
        if self.schedule == "1f1b":
            return self._build_1f1b()
        raise ValueError(f"unsupported schedule: {self.schedule}")

    def _build_gpipe(self) -> List[ScheduleAction]:
        actions: List[ScheduleAction] = []
        for i in range(self.num_micro_batches):
            actions.append(ScheduleAction("F", i))
        for i in reversed(range(self.num_micro_batches)):
            actions.append(ScheduleAction("B", i))
        return actions

    def _build_1f1b(self) -> List[ScheduleAction]:
        if self.pipeline_depth <= 1:
            return self._build_gpipe()

        actions: List[ScheduleAction] = []
        warmup = min(
            self.num_micro_batches,
            max(self.pipeline_depth - self.stage_index - 1, 0),
        )

        for i in range(warmup):
            actions.append(ScheduleAction("F", i))

        fwd_idx = warmup
        bwd_idx = 0
        remaining = self.num_micro_batches - warmup
        for _ in range(remaining):
            actions.append(ScheduleAction("F", fwd_idx))
            actions.append(ScheduleAction("B", bwd_idx))
            fwd_idx += 1
            bwd_idx += 1

        # Cooldown is needed for some stage positions when warmup > 0.
        for i in range(bwd_idx, self.num_micro_batches):
            actions.append(ScheduleAction("B", i))

        return actions


def theoretical_bubble_ratio(pipeline_depth: int, num_micro_batches: int) -> float:
    if pipeline_depth <= 1:
        return 0.0
    return float(pipeline_depth - 1) / float(num_micro_batches + pipeline_depth - 1)
