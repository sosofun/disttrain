from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.distributed as dist
from torch.optim import Optimizer


@dataclass
class _ParamSlice:
    param: torch.nn.Parameter
    start: int
    end: int


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class Zero1AdamW(Optimizer):
    """
    Distributed optimizer (ZeRO-1 style) with contiguous buffers:
    1) copy model grads -> fp32 main grad buffer
    2) reduce-scatter main grad buffer across DP ranks
    3) local AdamW update on fp32 main parameter shard
    4) cast local fp32 shard -> model dtype param shard
    5) all-gather updated param shards
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

        if len(self.param_groups) != 1:
            raise ValueError("Zero1AdamW currently supports a single parameter group only")
        if dp_global_ranks is not None:
            # Kept for compatibility; distributed collectives use dp_group directly.
            _ = dp_global_ranks

        self.dp_group = dp_group
        self._distributed = (
            dp_group is not None and dist.is_available() and dist.is_initialized()
        )
        self.dp_world_size = dist.get_world_size(dp_group) if self._distributed else 1
        self.dp_rank = dist.get_rank(dp_group) if self._distributed else 0

        self._params: List[torch.nn.Parameter] = list(self.param_groups[0]["params"])
        if not self._params:
            raise ValueError("Zero1AdamW received no parameters")

        self._validate_params(self._params)
        self._device = self._params[0].device
        self._param_dtype = self._params[0].dtype
        self._param_slices: List[_ParamSlice] = []
        self._numel = sum(p.numel() for p in self._params)

        self._shard_size = _ceil_div(self._numel, self.dp_world_size)
        self._padded_numel = self._shard_size * self.dp_world_size
        self._local_start = self.dp_rank * self._shard_size
        self._local_end = self._local_start + self._shard_size

        # Contiguous buffers.
        self._param_buffer_padded = torch.zeros(
            self._padded_numel,
            device=self._device,
            dtype=self._param_dtype,
        )
        self._main_param_buffer_padded = torch.zeros(
            self._padded_numel,
            device=self._device,
            dtype=torch.float32,
        )
        self._main_grad_buffer_padded = torch.zeros(
            self._padded_numel,
            device=self._device,
            dtype=torch.float32,
        )
        self._main_grad_mask_padded = torch.zeros(
            self._padded_numel,
            device=self._device,
            dtype=torch.bool,
        )

        self._param_buffer = self._param_buffer_padded[: self._numel]
        self._main_param_buffer = self._main_param_buffer_padded[: self._numel]
        self._main_grad_buffer = self._main_grad_buffer_padded[: self._numel]
        self._main_grad_mask = self._main_grad_mask_padded[: self._numel]

        # Build parameter views into contiguous parameter buffer.
        offset = 0
        for p in self._params:
            n = p.numel()
            flat = self._param_buffer[offset : offset + n]
            flat.copy_(p.data.reshape(-1))
            p.data = flat.view_as(p)
            self._param_slices.append(_ParamSlice(param=p, start=offset, end=offset + n))
            offset += n

        self._main_param_buffer.copy_(self._param_buffer.to(torch.float32))

        self._local_main_param_shard = self._main_param_buffer_padded[
            self._local_start : self._local_end
        ]
        self._local_param_shard = self._param_buffer_padded[self._local_start : self._local_end]
        self._local_main_grad_shard = torch.zeros(
            self._shard_size, device=self._device, dtype=torch.float32
        )
        self._local_grad_valid_mask = torch.zeros(
            self._shard_size, device=self._device, dtype=torch.bool
        )

        # Adam states are sharded (fp32).
        self._exp_avg_shard = torch.zeros(
            self._shard_size, device=self._device, dtype=torch.float32
        )
        self._exp_avg_sq_shard = torch.zeros(
            self._shard_size, device=self._device, dtype=torch.float32
        )
        self._step = 0
        self.zero_stage = 1
        self.last_sync_stats = {"time_sec": 0.0, "bytes_mb": 0.0}

    @staticmethod
    def _validate_params(params: List[torch.nn.Parameter]) -> None:
        device = params[0].device
        dtype = params[0].dtype
        for p in params:
            if not p.is_floating_point():
                raise ValueError("Zero1AdamW only supports floating-point parameters")
            if p.device != device:
                raise ValueError("Zero1AdamW requires all params on the same device")
            if p.dtype != dtype:
                raise ValueError("Zero1AdamW requires all params to have identical dtype")

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        super().zero_grad(set_to_none=set_to_none)
        self._main_grad_buffer.zero_()
        self._main_grad_buffer_padded[self._numel :].zero_()
        self._main_grad_mask.zero_()
        self._main_grad_mask_padded[self._numel :] = False
        self._local_main_grad_shard.zero_()
        self._local_grad_valid_mask.zero_()

    def _copy_model_grads_to_main_buffer(self) -> None:
        self._main_grad_buffer.zero_()
        self._main_grad_mask.zero_()
        for item in self._param_slices:
            grad = item.param.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError("Zero1AdamW does not support sparse gradients")
            self._main_grad_buffer[item.start : item.end].copy_(
                grad.detach().reshape(-1).to(torch.float32)
            )
            self._main_grad_mask[item.start : item.end] = True
            # Release model grad memory after copy to fp32 main gradient buffer.
            item.param.grad = None

    def _reduce_scatter_main_grads(self) -> Dict[str, float]:
        stats = {"time_sec": 0.0, "bytes_mb": 0.0}
        self._local_grad_valid_mask.copy_(
            self._main_grad_mask_padded[self._local_start : self._local_end]
        )
        if self.dp_world_size <= 1 or not self._distributed:
            self._local_main_grad_shard.copy_(
                self._main_grad_buffer_padded[self._local_start : self._local_end]
            )
            return stats

        t0 = time.perf_counter()
        used_fallback = False
        if hasattr(dist, "reduce_scatter_tensor"):
            try:
                dist.reduce_scatter_tensor(
                    output=self._local_main_grad_shard,
                    input=self._main_grad_buffer_padded,
                    op=dist.ReduceOp.SUM,
                    group=self.dp_group,
                )
            except Exception:
                used_fallback = True
        else:
            used_fallback = True
        if used_fallback:
            chunks = list(self._main_grad_buffer_padded.chunk(self.dp_world_size))
            dist.reduce_scatter(
                output=self._local_main_grad_shard,
                input_list=chunks,
                op=dist.ReduceOp.SUM,
                group=self.dp_group,
            )
        self._local_main_grad_shard /= float(self.dp_world_size)
        elapsed = time.perf_counter() - t0
        stats["time_sec"] += elapsed
        stats["bytes_mb"] += float(self._padded_numel * 4) / (1024.0 * 1024.0)
        return stats

    def _local_adamw_update(self) -> None:
        group = self.param_groups[0]
        lr = float(group["lr"])
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        weight_decay = float(group["weight_decay"])

        mask = self._local_grad_valid_mask
        if not bool(mask.any()):
            return

        grad = self._local_main_grad_shard
        old_exp_avg = self._exp_avg_shard
        old_exp_avg_sq = self._exp_avg_sq_shard

        exp_avg_new = old_exp_avg * beta1 + grad * (1.0 - beta1)
        exp_avg_sq_new = old_exp_avg_sq * beta2 + grad * grad * (1.0 - beta2)
        self._exp_avg_shard.copy_(torch.where(mask, exp_avg_new, old_exp_avg))
        self._exp_avg_sq_shard.copy_(torch.where(mask, exp_avg_sq_new, old_exp_avg_sq))

        bias_correction1 = 1.0 - beta1**self._step
        bias_correction2 = 1.0 - beta2**self._step
        step_size = lr * (bias_correction2**0.5) / bias_correction1

        denom = exp_avg_sq_new.sqrt().add_(eps)
        update = exp_avg_new / denom
        updated_param = self._local_main_param_shard
        if weight_decay != 0.0:
            updated_param = updated_param * (1.0 - lr * weight_decay)
        updated_param = updated_param - step_size * update
        self._local_main_param_shard.copy_(
            torch.where(mask, updated_param, self._local_main_param_shard)
        )

    def _all_gather_updated_params(self) -> Dict[str, float]:
        stats = {"time_sec": 0.0, "bytes_mb": 0.0}
        # Copy local updated fp32 shard -> model dtype shard.
        self._local_param_shard.copy_(self._local_main_param_shard.to(self._param_dtype))

        if self.dp_world_size <= 1 or not self._distributed:
            return stats

        t0 = time.perf_counter()
        used_fallback = False
        if hasattr(dist, "all_gather_into_tensor"):
            try:
                dist.all_gather_into_tensor(
                    output_tensor=self._param_buffer_padded,
                    input_tensor=self._local_param_shard,
                    group=self.dp_group,
                )
            except Exception:
                used_fallback = True
        else:
            used_fallback = True
        if used_fallback:
            gather_chunks = list(self._param_buffer_padded.chunk(self.dp_world_size))
            dist.all_gather(gather_chunks, self._local_param_shard, group=self.dp_group)
        elapsed = time.perf_counter() - t0
        stats["time_sec"] += elapsed
        stats["bytes_mb"] += float(self._padded_numel * self._local_param_shard.element_size()) / (
            1024.0 * 1024.0
        )
        return stats

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step += 1
        self._copy_model_grads_to_main_buffer()
        rs_stats = self._reduce_scatter_main_grads()
        self._local_adamw_update()
        ag_stats = self._all_gather_updated_params()

        self._main_grad_buffer.zero_()
        self._main_grad_buffer_padded[self._numel :].zero_()
        self._main_grad_mask.zero_()
        self._main_grad_mask_padded[self._numel :] = False

        self.last_sync_stats = {
            "time_sec": float(rs_stats["time_sec"] + ag_stats["time_sec"]),
            "bytes_mb": float(rs_stats["bytes_mb"] + ag_stats["bytes_mb"]),
        }
        return loss

    def state_dict(self) -> Dict[str, object]:  # type: ignore[override]
        base = super().state_dict()
        base["dist_optim"] = {
            "step": int(self._step),
            "numel": int(self._numel),
            "padded_numel": int(self._padded_numel),
            "shard_size": int(self._shard_size),
            "main_param_shard": self._local_main_param_shard.detach().cpu(),
            "exp_avg_shard": self._exp_avg_shard.detach().cpu(),
            "exp_avg_sq_shard": self._exp_avg_sq_shard.detach().cpu(),
        }
        return base

    def load_state_dict(self, state_dict: Dict[str, object]) -> None:  # type: ignore[override]
        dist_state = state_dict.get("dist_optim")
        super().load_state_dict(state_dict)

        if not isinstance(dist_state, dict):
            # Fallback: rebuild fp32 main params from model param buffer.
            self._main_param_buffer.copy_(self._param_buffer.to(torch.float32))
            self._exp_avg_shard.zero_()
            self._exp_avg_sq_shard.zero_()
            self._step = 0
            return

        self._step = int(dist_state.get("step", 0))
        for key, target in (
            ("main_param_shard", self._local_main_param_shard),
            ("exp_avg_shard", self._exp_avg_shard),
            ("exp_avg_sq_shard", self._exp_avg_sq_shard),
        ):
            loaded = dist_state.get(key)
            if loaded is None:
                continue
            tensor = torch.as_tensor(loaded, device=target.device, dtype=target.dtype)
            if tensor.numel() != target.numel():
                raise ValueError(
                    f"optimizer shard shape mismatch for {key}: "
                    f"ckpt={tensor.numel()}, runtime={target.numel()}"
                )
            target.copy_(tensor.view_as(target))

        self._local_param_shard.copy_(self._local_main_param_shard.to(self._param_dtype))
        if self.dp_world_size > 1 and self._distributed:
            used_fallback = False
            if hasattr(dist, "all_gather_into_tensor"):
                try:
                    dist.all_gather_into_tensor(
                        output_tensor=self._param_buffer_padded,
                        input_tensor=self._local_param_shard,
                        group=self.dp_group,
                    )
                except Exception:
                    used_fallback = True
            else:
                used_fallback = True
            if used_fallback:
                gather_chunks = list(self._param_buffer_padded.chunk(self.dp_world_size))
                dist.all_gather(gather_chunks, self._local_param_shard, group=self.dp_group)


# Backward-compatible alias: current ZeRO-1 implementation is a distributed optimizer.
DistributedOptimizer = Zero1AdamW
