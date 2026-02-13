from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from disttrain.dist.topology import Topology


def _is_dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


@dataclass
class PeerRoute:
    from_stage: str
    to_stage: str
    from_rank: int
    to_rank: int


class P2PRouter:
    """
    A simple boundary router:
    - cross-stage p2p uses tp_idx=0 ranks for transport.
    - dp mapping currently follows modulo route (compatible with VPS mapping).
    """

    def __init__(self, topology: Topology):
        self.topology = topology

    def next_peer_rank(self, stage_name: str, dp_idx: int) -> Optional[int]:
        next_stage = self.topology.next_stage(stage_name)
        if next_stage is None:
            return None
        next_dp_size = self.topology.stages[next_stage].dp_size
        next_dp = dp_idx % next_dp_size
        return self.topology.rank_for(next_stage, next_dp, 0)

    def prev_peer_rank(self, stage_name: str, dp_idx: int) -> Optional[int]:
        prev_stage = self.topology.prev_stage(stage_name)
        if prev_stage is None:
            return None
        prev_dp_size = self.topology.stages[prev_stage].dp_size
        prev_dp = dp_idx % prev_dp_size
        return self.topology.rank_for(prev_stage, prev_dp, 0)

    def route_to_next(self, stage_name: str, dp_idx: int) -> Optional[PeerRoute]:
        next_stage = self.topology.next_stage(stage_name)
        if next_stage is None:
            return None
        src_rank = self.topology.rank_for(stage_name, dp_idx, 0)
        dst_rank = self.next_peer_rank(stage_name, dp_idx)
        if dst_rank is None:
            return None
        return PeerRoute(stage_name, next_stage, src_rank, dst_rank)

    def route_to_prev(self, stage_name: str, dp_idx: int) -> Optional[PeerRoute]:
        prev_stage = self.topology.prev_stage(stage_name)
        if prev_stage is None:
            return None
        src_rank = self.topology.rank_for(stage_name, dp_idx, 0)
        dst_rank = self.prev_peer_rank(stage_name, dp_idx)
        if dst_rank is None:
            return None
        return PeerRoute(stage_name, prev_stage, src_rank, dst_rank)


def activation_tag(step: int, micro_batch_idx: int) -> int:
    return step * 10000 + 1000 + micro_batch_idx


def gradient_tag(step: int, micro_batch_idx: int) -> int:
    return step * 10000 + 5000 + micro_batch_idx


def send_tensor(
    tensor: torch.Tensor,
    dst_rank: int,
    tag: int,
    async_op: bool = False,
) -> Optional[Tuple[dist.Work, torch.Tensor]]:
    if not _is_dist_ready():
        return None
    payload = tensor.detach().contiguous()
    if async_op:
        # Keep a dedicated payload buffer to avoid lifetime issues when the
        # original tensor is reused or released before async send completion.
        payload = payload.clone()
        work = dist.isend(tensor=payload, dst=dst_rank, tag=tag)
        return work, payload
    dist.send(tensor=payload, dst=dst_rank, tag=tag)
    return None


def recv_tensor(
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    src_rank: int,
    tag: int,
    async_op: bool = False,
) -> Tuple[torch.Tensor, Optional[dist.Work]]:
    tensor = torch.empty(shape, dtype=dtype, device=device)
    if not _is_dist_ready():
        return tensor, None
    if async_op:
        work = dist.irecv(tensor=tensor, src=src_rank, tag=tag)
        return tensor, work
    dist.recv(tensor=tensor, src=src_rank, tag=tag)
    return tensor, None
