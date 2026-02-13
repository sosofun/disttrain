from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist

from disttrain.dist.topology import Topology


class ProcessGroupManager:
    """
    Manages stage/tp/dp process groups.
    """

    def __init__(self, topology: Topology):
        self.topology = topology
        self.stage_groups: Dict[str, dist.ProcessGroup] = {}
        self.tp_groups: Dict[Tuple[str, int], dist.ProcessGroup] = {}
        self.dp_groups: Dict[Tuple[str, int], dist.ProcessGroup] = {}
        self.boundary_groups: Dict[Tuple[str, str, int], dist.ProcessGroup] = {}

        self.local_stage_group: Optional[dist.ProcessGroup] = None
        self.local_tp_group: Optional[dist.ProcessGroup] = None
        self.local_dp_group: Optional[dist.ProcessGroup] = None

    @property
    def is_initialized(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    def create(self) -> None:
        if not self.is_initialized:
            return

        for stage_name in self.topology.enabled_stage_names:
            stage = self.topology.stages[stage_name]
            stage_ranks = list(range(stage.start_rank, stage.end_rank + 1))
            self.stage_groups[stage_name] = dist.new_group(ranks=stage_ranks)

            for dp_idx in range(stage.dp_size):
                tp_ranks = [
                    self.topology.rank_for(stage_name, dp_idx, tp_idx)
                    for tp_idx in range(stage.tp_size)
                ]
                self.tp_groups[(stage_name, dp_idx)] = dist.new_group(ranks=tp_ranks)

            for tp_idx in range(stage.tp_size):
                dp_ranks = [
                    self.topology.rank_for(stage_name, dp_idx, tp_idx)
                    for dp_idx in range(stage.dp_size)
                ]
                self.dp_groups[(stage_name, tp_idx)] = dist.new_group(ranks=dp_ranks)

        enabled = self.topology.enabled_stage_names
        for i in range(len(enabled) - 1):
            from_stage = enabled[i]
            to_stage = enabled[i + 1]
            from_dp = self.topology.stages[from_stage].dp_size
            to_dp = self.topology.stages[to_stage].dp_size
            vps_size = math.lcm(from_dp, to_dp)
            for vps in range(vps_size):
                src_rank = self.topology.rank_for(from_stage, vps % from_dp, 0)
                dst_rank = self.topology.rank_for(to_stage, vps % to_dp, 0)
                ranks = sorted(set([src_rank, dst_rank]))
                self.boundary_groups[(from_stage, to_stage, vps)] = dist.new_group(
                    ranks=ranks
                )

        local_stage = self.topology.local_stage_name
        local_dp = self.topology.local_dp_index()
        local_tp = self.topology.local_tp_index()
        self.local_stage_group = self.stage_groups[local_stage]
        self.local_tp_group = self.tp_groups[(local_stage, local_dp)]
        self.local_dp_group = self.dp_groups[(local_stage, local_tp)]

    def stage_group(self, stage_name: str) -> Optional[dist.ProcessGroup]:
        return self.stage_groups.get(stage_name)

    def tp_group(self, stage_name: str, dp_idx: int) -> Optional[dist.ProcessGroup]:
        return self.tp_groups.get((stage_name, dp_idx))

    def dp_group(self, stage_name: str, tp_idx: int) -> Optional[dist.ProcessGroup]:
        return self.dp_groups.get((stage_name, tp_idx))

    def sync_parameters(self, model: torch.nn.Module) -> None:
        """
        Make TP replicas start from same parameters and then keep DP replicas in sync.
        """
        if not self.is_initialized:
            return

        local_stage = self.topology.local_stage_name
        local_dp = self.topology.local_dp_index()
        local_tp = self.topology.local_tp_index()
        stage = self.topology.stages[local_stage]

        tp_src_rank = self.topology.rank_for(local_stage, local_dp, 0)
        dp_src_rank = self.topology.rank_for(local_stage, 0, local_tp)

        tp_group = self.tp_group(local_stage, local_dp)
        dp_group = self.dp_group(local_stage, local_tp)
        if tp_group is None or dp_group is None:
            return

        for p in model.parameters():
            dist.broadcast(p.data, src=tp_src_rank, group=tp_group)
            dist.broadcast(p.data, src=dp_src_rank, group=dp_group)

        # Keep buffers (e.g., LayerNorm running stats if any) aligned as well.
        for b in model.buffers():
            dist.broadcast(b.data, src=tp_src_rank, group=tp_group)
            dist.broadcast(b.data, src=dp_src_rank, group=dp_group)

        if stage.tp_size == 1 and stage.dp_size == 1:
            return

    def average_gradients(self, model: torch.nn.Module) -> None:
        if not self.is_initialized:
            return

        local_stage = self.topology.local_stage_name
        stage = self.topology.stages[local_stage]
        tp_group = self.local_tp_group
        dp_group = self.local_dp_group
        if tp_group is None or dp_group is None:
            return

        for p in model.parameters():
            if p.grad is None:
                continue
            if stage.tp_size > 1:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=tp_group)
                p.grad /= float(stage.tp_size)
            if stage.dp_size > 1:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=dp_group)
                p.grad /= float(stage.dp_size)

    def clear(self) -> None:
        self.stage_groups.clear()
        self.tp_groups.clear()
        self.dp_groups.clear()
        self.boundary_groups.clear()
        self.local_stage_group = None
        self.local_tp_group = None
        self.local_dp_group = None
