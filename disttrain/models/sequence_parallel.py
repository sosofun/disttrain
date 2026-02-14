from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist


def _require_divisible(seq_len: int, tp_size: int) -> None:
    if seq_len % tp_size != 0:
        raise ValueError(
            f"sequence length {seq_len} must be divisible by tp_size {tp_size}"
        )


def _is_dist(group: Optional[dist.ProcessGroup], tp_size: int) -> bool:
    return (
        tp_size > 1
        and group is not None
        and dist.is_available()
        and dist.is_initialized()
    )


def split_sequence_local(
    tensor: torch.Tensor,
    tp_size: int,
    tp_rank: int,
) -> torch.Tensor:
    if tp_size <= 1:
        return tensor
    _require_divisible(tensor.size(1), tp_size)
    chunk = tensor.size(1) // tp_size
    start = tp_rank * chunk
    end = start + chunk
    return tensor[:, start:end, :].contiguous()


class _SequenceAllGather(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        group: Optional[dist.ProcessGroup],
        tp_size: int,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.tp_size = tp_size
        if not _is_dist(group, tp_size):
            return x
        gathered = [torch.empty_like(x) for _ in range(tp_size)]
        dist.all_gather(gathered, x, group=group)
        return torch.cat(gathered, dim=1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:  # type: ignore[override]
        group = ctx.group
        tp_size = ctx.tp_size
        if not _is_dist(group, tp_size):
            return grad_output, None, None

        _require_divisible(grad_output.size(1), tp_size)
        chunks = [t.contiguous() for t in grad_output.chunk(tp_size, dim=1)]
        out = torch.empty_like(chunks[0])
        dist.reduce_scatter(out, chunks, op=dist.ReduceOp.SUM, group=group)
        return out, None, None


class _SequenceReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        group: Optional[dist.ProcessGroup],
        tp_size: int,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.tp_size = tp_size
        if not _is_dist(group, tp_size):
            return x

        _require_divisible(x.size(1), tp_size)
        chunks = [t.contiguous() for t in x.chunk(tp_size, dim=1)]
        out = torch.empty_like(chunks[0])
        dist.reduce_scatter(out, chunks, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:  # type: ignore[override]
        group = ctx.group
        tp_size = ctx.tp_size
        if not _is_dist(group, tp_size):
            return grad_output, None, None

        gathered = [torch.empty_like(grad_output) for _ in range(tp_size)]
        dist.all_gather(gathered, grad_output, group=group)
        return torch.cat(gathered, dim=1), None, None


def sequence_all_gather(
    x: torch.Tensor,
    group: Optional[dist.ProcessGroup],
    tp_size: int,
) -> torch.Tensor:
    return _SequenceAllGather.apply(x, group, tp_size)


def sequence_reduce_scatter(
    x: torch.Tensor,
    group: Optional[dist.ProcessGroup],
    tp_size: int,
) -> torch.Tensor:
    return _SequenceReduceScatter.apply(x, group, tp_size)
