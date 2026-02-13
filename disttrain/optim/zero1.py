from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

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


class _BufferBucket:
    def __init__(
        self,
        *,
        dtype: torch.dtype,
        params: List[torch.nn.Parameter],
        dp_world_size: int,
        dp_rank: int,
        device: torch.device,
    ):
        self.dtype = dtype
        self.params = params
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.device = device
        self.param_slices: List[_ParamSlice] = []

        self.numel = sum(p.numel() for p in self.params)
        self.shard_size = _ceil_div(self.numel, self.dp_world_size)
        self.padded_numel = self.shard_size * self.dp_world_size
        self.local_start = self.dp_rank * self.shard_size
        self.local_end = self.local_start + self.shard_size

        self.param_buffer_padded = torch.zeros(
            self.padded_numel,
            device=self.device,
            dtype=self.dtype,
        )
        self.main_param_buffer_padded = torch.zeros(
            self.padded_numel,
            device=self.device,
            dtype=torch.float32,
        )
        self.main_grad_buffer_padded = torch.zeros(
            self.padded_numel,
            device=self.device,
            dtype=torch.float32,
        )

        self.param_buffer = self.param_buffer_padded[: self.numel]
        self.main_param_buffer = self.main_param_buffer_padded[: self.numel]
        self.main_grad_buffer = self.main_grad_buffer_padded[: self.numel]

        offset = 0
        for p in self.params:
            n = p.numel()
            flat = self.param_buffer[offset : offset + n]
            flat.copy_(p.data.reshape(-1))
            p.data = flat.view_as(p)
            self.param_slices.append(_ParamSlice(param=p, start=offset, end=offset + n))
            offset += n

        self.main_param_buffer.copy_(self.param_buffer.to(torch.float32))

        self.local_main_param_shard = self.main_param_buffer_padded[
            self.local_start : self.local_end
        ]
        self.local_param_shard = self.param_buffer_padded[self.local_start : self.local_end]
        self.local_main_grad_shard = torch.zeros(
            self.shard_size, device=self.device, dtype=torch.float32
        )
        self.local_active_ranges: List[Tuple[int, int]] = []

        self.exp_avg_shard = torch.zeros(
            self.shard_size, device=self.device, dtype=torch.float32
        )
        self.exp_avg_sq_shard = torch.zeros(
            self.shard_size, device=self.device, dtype=torch.float32
        )

    def _mark_local_range_active(self, global_start: int, global_end: int) -> None:
        start = max(global_start, self.local_start)
        end = min(global_end, self.local_end)
        if start >= end:
            return
        local_start = start - self.local_start
        local_end = end - self.local_start
        if self.local_active_ranges and self.local_active_ranges[-1][1] == local_start:
            prev_start, _prev_end = self.local_active_ranges[-1]
            self.local_active_ranges[-1] = (prev_start, local_end)
            return
        self.local_active_ranges.append((local_start, local_end))

    def zero_grad_buffers(self) -> None:
        self.main_grad_buffer.zero_()
        self.main_grad_buffer_padded[self.numel :].zero_()
        self.local_main_grad_shard.zero_()
        self.local_active_ranges.clear()

    def copy_model_grads_to_main_buffer(self) -> None:
        self.main_grad_buffer.zero_()
        self.main_grad_buffer_padded[self.numel :].zero_()
        self.local_active_ranges.clear()
        for item in self.param_slices:
            grad = item.param.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError("Zero1AdamW does not support sparse gradients")
            self.main_grad_buffer[item.start : item.end].copy_(
                grad.detach().reshape(-1).to(torch.float32)
            )
            self._mark_local_range_active(item.start, item.end)
            item.param.grad = None

    def reduce_scatter_main_grads(
        self,
        *,
        distributed: bool,
        dp_group: Optional[dist.ProcessGroup],
    ) -> Dict[str, float]:
        stats = {"time_sec": 0.0, "bytes_mb": 0.0}
        if self.dp_world_size <= 1 or not distributed:
            self.local_main_grad_shard.copy_(
                self.main_grad_buffer_padded[self.local_start : self.local_end]
            )
            return stats

        t0 = time.perf_counter()
        used_fallback = False
        if hasattr(dist, "reduce_scatter_tensor"):
            try:
                dist.reduce_scatter_tensor(
                    output=self.local_main_grad_shard,
                    input=self.main_grad_buffer_padded,
                    op=dist.ReduceOp.SUM,
                    group=dp_group,
                )
            except Exception:
                used_fallback = True
        else:
            used_fallback = True
        if used_fallback:
            chunks = list(self.main_grad_buffer_padded.chunk(self.dp_world_size))
            dist.reduce_scatter(
                output=self.local_main_grad_shard,
                input_list=chunks,
                op=dist.ReduceOp.SUM,
                group=dp_group,
            )
        self.local_main_grad_shard /= float(self.dp_world_size)
        elapsed = time.perf_counter() - t0
        stats["time_sec"] += elapsed
        stats["bytes_mb"] += float(self.padded_numel * 4) / (1024.0 * 1024.0)
        return stats

    def local_adamw_update(
        self,
        *,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        step: int,
    ) -> None:
        if not self.local_active_ranges:
            return

        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        step_size = lr * (bias_correction2**0.5) / bias_correction1

        for start, end in self.local_active_ranges:
            grad = self.local_main_grad_shard[start:end]
            exp_avg = self.exp_avg_shard[start:end]
            exp_avg_sq = self.exp_avg_sq_shard[start:end]
            param = self.local_main_param_shard[start:end]

            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            denom = exp_avg_sq.sqrt().add_(eps)
            if weight_decay != 0.0:
                param.mul_(1.0 - lr * weight_decay)
            param.addcdiv_(exp_avg, denom, value=-step_size)

    def _all_gather_param_buffer(
        self,
        *,
        dp_group: Optional[dist.ProcessGroup],
    ) -> None:
        used_fallback = False
        if hasattr(dist, "all_gather_into_tensor"):
            try:
                dist.all_gather_into_tensor(
                    output_tensor=self.param_buffer_padded,
                    input_tensor=self.local_param_shard,
                    group=dp_group,
                )
            except Exception:
                used_fallback = True
        else:
            used_fallback = True
        if used_fallback:
            gather_chunks = list(self.param_buffer_padded.chunk(self.dp_world_size))
            dist.all_gather(gather_chunks, self.local_param_shard, group=dp_group)

    def all_gather_updated_params(
        self,
        *,
        distributed: bool,
        dp_group: Optional[dist.ProcessGroup],
    ) -> Dict[str, float]:
        stats = {"time_sec": 0.0, "bytes_mb": 0.0}
        self.local_param_shard.copy_(self.local_main_param_shard.to(self.dtype))
        if self.dp_world_size <= 1 or not distributed:
            return stats

        t0 = time.perf_counter()
        self._all_gather_param_buffer(dp_group=dp_group)
        elapsed = time.perf_counter() - t0
        stats["time_sec"] += elapsed
        stats["bytes_mb"] += float(self.padded_numel * self.local_param_shard.element_size()) / (
            1024.0 * 1024.0
        )
        return stats

    def export_state(self) -> Dict[str, object]:
        return {
            "dtype": str(self.dtype),
            "numel": int(self.numel),
            "padded_numel": int(self.padded_numel),
            "shard_size": int(self.shard_size),
            "main_param_shard": self.local_main_param_shard.detach().cpu(),
            "exp_avg_shard": self.exp_avg_shard.detach().cpu(),
            "exp_avg_sq_shard": self.exp_avg_sq_shard.detach().cpu(),
        }

    def rebuild_from_param_buffer(self) -> None:
        self.main_param_buffer.copy_(self.param_buffer.to(torch.float32))
        self.exp_avg_shard.zero_()
        self.exp_avg_sq_shard.zero_()

    def load_state(
        self,
        state: Dict[str, object],
        *,
        distributed: bool,
        dp_group: Optional[dist.ProcessGroup],
    ) -> None:
        for key, target in (
            ("main_param_shard", self.local_main_param_shard),
            ("exp_avg_shard", self.exp_avg_shard),
            ("exp_avg_sq_shard", self.exp_avg_sq_shard),
        ):
            loaded = state.get(key)
            if loaded is None:
                continue
            tensor = torch.as_tensor(loaded, device=target.device, dtype=target.dtype)
            if tensor.numel() != target.numel():
                raise ValueError(
                    f"optimizer shard shape mismatch for {key}: "
                    f"ckpt={tensor.numel()}, runtime={target.numel()}"
                )
            target.copy_(tensor.view_as(target))

        self.local_param_shard.copy_(self.local_main_param_shard.to(self.dtype))
        if self.dp_world_size > 1 and distributed:
            self._all_gather_param_buffer(dp_group=dp_group)


class Zero1AdamW(Optimizer):
    """
    Distributed optimizer (ZeRO-1 style) with contiguous buffers:
    1) copy model grads -> fp32 main grad buffer
    2) reduce-scatter main grad buffer across DP ranks
    3) local AdamW update on fp32 main parameter shard
    4) cast local fp32 shard -> model dtype param shard
    5) all-gather updated param shards

    This implementation groups model parameters by dtype (e.g., bf16/fp16/fp32)
    and maintains an independent contiguous buffer set for each dtype bucket.
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
        dtype_groups = self._group_params_by_dtype(self._params)
        self._buckets: List[_BufferBucket] = [
            _BufferBucket(
                dtype=dtype,
                params=params_of_dtype,
                dp_world_size=self.dp_world_size,
                dp_rank=self.dp_rank,
                device=self._device,
            )
            for dtype, params_of_dtype in dtype_groups
        ]
        self._step = 0
        self.zero_stage = 1
        self.last_sync_stats = {"time_sec": 0.0, "bytes_mb": 0.0}

    @staticmethod
    def _validate_params(params: List[torch.nn.Parameter]) -> None:
        device = params[0].device
        for p in params:
            if not p.is_floating_point():
                raise ValueError("Zero1AdamW only supports floating-point parameters")
            if p.device != device:
                raise ValueError("Zero1AdamW requires all params on the same device")

    @staticmethod
    def _group_params_by_dtype(
        params: List[torch.nn.Parameter],
    ) -> List[Tuple[torch.dtype, List[torch.nn.Parameter]]]:
        groups: Dict[torch.dtype, List[torch.nn.Parameter]] = {}
        order: List[torch.dtype] = []
        for p in params:
            if p.dtype not in groups:
                groups[p.dtype] = []
                order.append(p.dtype)
            groups[p.dtype].append(p)
        return [(dtype, groups[dtype]) for dtype in order]

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        super().zero_grad(set_to_none=set_to_none)
        for bucket in self._buckets:
            bucket.zero_grad_buffers()

    def _step_hyperparams(self) -> Tuple[float, float, float, float, float]:
        group = self.param_groups[0]
        lr = float(group["lr"])
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        weight_decay = float(group["weight_decay"])
        return lr, float(beta1), float(beta2), eps, weight_decay

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step += 1
        lr, beta1, beta2, eps, weight_decay = self._step_hyperparams()

        total_rs_time = 0.0
        total_rs_mb = 0.0
        total_ag_time = 0.0
        total_ag_mb = 0.0
        for bucket in self._buckets:
            bucket.copy_model_grads_to_main_buffer()
            rs_stats = bucket.reduce_scatter_main_grads(
                distributed=self._distributed,
                dp_group=self.dp_group,
            )
            bucket.local_adamw_update(
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                weight_decay=weight_decay,
                step=self._step,
            )
            ag_stats = bucket.all_gather_updated_params(
                distributed=self._distributed,
                dp_group=self.dp_group,
            )
            bucket.zero_grad_buffers()
            total_rs_time += float(rs_stats["time_sec"])
            total_rs_mb += float(rs_stats["bytes_mb"])
            total_ag_time += float(ag_stats["time_sec"])
            total_ag_mb += float(ag_stats["bytes_mb"])

        self.last_sync_stats = {
            "time_sec": float(total_rs_time + total_ag_time),
            "bytes_mb": float(total_rs_mb + total_ag_mb),
        }
        return loss

    def state_dict(self) -> Dict[str, object]:  # type: ignore[override]
        base = super().state_dict()
        base["dist_optim"] = {
            "step": int(self._step),
            "buckets": [bucket.export_state() for bucket in self._buckets],
        }
        return base

    def load_state_dict(self, state_dict: Dict[str, object]) -> None:  # type: ignore[override]
        dist_state = state_dict.get("dist_optim")
        super().load_state_dict(state_dict)

        if not isinstance(dist_state, dict):
            for bucket in self._buckets:
                bucket.rebuild_from_param_buffer()
            self._step = 0
            return

        self._step = int(dist_state.get("step", 0))
        saved_buckets = dist_state.get("buckets")
        if isinstance(saved_buckets, list):
            saved_map: Dict[Tuple[str, int], Dict[str, object]] = {}
            for item in saved_buckets:
                if not isinstance(item, dict):
                    continue
                key = (str(item.get("dtype")), int(item.get("numel", -1)))
                saved_map[key] = item

            for bucket in self._buckets:
                key = (str(bucket.dtype), int(bucket.numel))
                saved = saved_map.get(key)
                if saved is None:
                    raise ValueError(
                        "checkpoint optimizer bucket mismatch for dtype/numel: "
                        f"dtype={bucket.dtype}, numel={bucket.numel}"
                    )
                bucket.load_state(
                    saved,
                    distributed=self._distributed,
                    dp_group=self.dp_group,
                )
            return

        # Backward compatibility: old single-bucket format.
        if len(self._buckets) == 1:
            legacy = {
                "main_param_shard": dist_state.get("main_param_shard"),
                "exp_avg_shard": dist_state.get("exp_avg_shard"),
                "exp_avg_sq_shard": dist_state.get("exp_avg_sq_shard"),
            }
            self._buckets[0].load_state(
                legacy,
                distributed=self._distributed,
                dp_group=self.dp_group,
            )
            return

        raise ValueError(
            "checkpoint optimizer state format is incompatible with mixed-dtype buckets"
        )


# Backward-compatible alias: current ZeRO-1 implementation is a distributed optimizer.
DistributedOptimizer = Zero1AdamW
