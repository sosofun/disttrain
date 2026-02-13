from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional

import torch
import torch.distributed as dist
from torch.optim import Optimizer


def _build_balanced_owners(
    params: List[torch.nn.Parameter],
    world_size: int,
) -> List[int]:
    if world_size <= 1:
        return [0 for _ in params]

    order = sorted(
        range(len(params)),
        key=lambda i: params[i].numel(),
        reverse=True,
    )
    load = [0 for _ in range(world_size)]
    owners = [0 for _ in params]
    for idx in order:
        owner = min(range(world_size), key=lambda r: load[r])
        owners[idx] = owner
        load[owner] += params[idx].numel()
    return owners


class Zero1AdamW(Optimizer):
    """
    ZeRO-1 optimizer: shard optimizer states across data-parallel ranks.
    Parameters/gradients remain replicated; only owners update their shards,
    then updated parameters are broadcast within the DP group.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        dp_group: Optional[dist.ProcessGroup] = None,
        dp_global_ranks: Optional[List[int]] = None,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.dp_group = dp_group
        self._distributed = (
            dp_group is not None and dist.is_available() and dist.is_initialized()
        )
        self.dp_world_size = dist.get_world_size(dp_group) if self._distributed else 1
        self.dp_rank = dist.get_rank(dp_group) if self._distributed else 0
        if dp_global_ranks is not None:
            self.dp_global_ranks = dp_global_ranks
        elif self._distributed:
            try:
                self.dp_global_ranks = [
                    dist.get_global_rank(dp_group, i) for i in range(self.dp_world_size)
                ]
            except Exception:
                # Fallback for environments where get_global_rank is unavailable.
                self.dp_global_ranks = [dist.get_rank() for _ in range(self.dp_world_size)]
        else:
            self.dp_global_ranks = [0]
        if len(self.dp_global_ranks) != self.dp_world_size:
            raise ValueError(
                "dp_global_ranks size mismatch: "
                f"len={len(self.dp_global_ranks)}, world={self.dp_world_size}"
            )

        self._param_list: List[torch.nn.Parameter] = [
            p for group in self.param_groups for p in group["params"]
        ]
        self._owners = _build_balanced_owners(self._param_list, self.dp_world_size)
        self._param_index = {id(p): i for i, p in enumerate(self._param_list)}
        self.zero_stage = 1
        self.last_sync_stats = {"time_sec": 0.0, "bytes_mb": 0.0}

    def _is_owner(self, p: torch.nn.Parameter) -> bool:
        idx = self._param_index[id(p)]
        return self._owners[idx] == self.dp_rank

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if not self._is_owner(p):
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Zero1AdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step_t = state["step"]

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(eps)

                bias_correction1 = 1 - beta1**step_t
                bias_correction2 = 1 - beta2**step_t
                step_size = lr * (bias_correction2**0.5) / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        self.last_sync_stats = self._broadcast_updated_parameters()
        return loss

    def _broadcast_updated_parameters(self) -> Dict[str, float]:
        stats = {"time_sec": 0.0, "bytes_mb": 0.0}
        if not self._distributed or self.dp_world_size <= 1:
            return stats

        t0 = time.perf_counter()
        total_bytes = 0
        for idx, p in enumerate(self._param_list):
            owner_dp_rank = self._owners[idx]
            src_global_rank = self.dp_global_ranks[owner_dp_rank]
            dist.broadcast(p.data, src=src_global_rank, group=self.dp_group)
            total_bytes += p.numel() * p.element_size()
        stats["time_sec"] = time.perf_counter() - t0
        stats["bytes_mb"] = float(total_bytes) / (1024.0 * 1024.0)
        return stats
