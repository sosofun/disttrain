from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _check_divisible(name: str, value: int, divisor: int) -> None:
    if value % divisor != 0:
        raise ValueError(f"{name}={value} must be divisible by tp_size={divisor}")


def _mark_tp_sharded(param: torch.nn.Parameter) -> None:
    setattr(param, "_tp_sharded", True)


def _all_reduce_autograd(
    tensor: torch.Tensor,
    group: Optional[dist.ProcessGroup],
) -> torch.Tensor:
    if group is None or not (dist.is_available() and dist.is_initialized()):
        return tensor
    try:
        from torch.distributed.nn import functional as dist_nn_f  # type: ignore

        return dist_nn_f.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    except Exception:
        out = tensor.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out


def _all_gather_autograd(
    tensor: torch.Tensor,
    group: Optional[dist.ProcessGroup],
    world_size: int,
) -> torch.Tensor:
    if world_size == 1 or group is None or not (dist.is_available() and dist.is_initialized()):
        return tensor
    try:
        from torch.distributed.nn import functional as dist_nn_f  # type: ignore

        gathered = dist_nn_f.all_gather(tensor, group=group)
        if isinstance(gathered, (tuple, list)):
            return torch.cat(list(gathered), dim=-1)
        return gathered
    except Exception:
        gathered_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_list, tensor, group=group)
        return torch.cat(gathered_list, dim=-1)


class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tp_size: int,
        tp_rank: int,
    ):
        super().__init__()
        _check_divisible("num_embeddings", num_embeddings, tp_size)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group: Optional[dist.ProcessGroup] = None

        self.local_vocab_size = num_embeddings // tp_size
        self.vocab_start = tp_rank * self.local_vocab_size
        self.vocab_end = self.vocab_start + self.local_vocab_size
        self.weight = nn.Parameter(torch.empty(self.local_vocab_size, embedding_dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        _mark_tp_sharded(self.weight)

    def set_tp_group(self, group: Optional[dist.ProcessGroup]) -> None:
        self.tp_group = group

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return F.embedding(input_ids, self.weight)

        mask = (input_ids < self.vocab_start) | (input_ids >= self.vocab_end)
        local_ids = input_ids - self.vocab_start
        local_ids = local_ids.clamp(min=0, max=self.local_vocab_size - 1)
        out = F.embedding(local_ids, self.weight)
        out = out.masked_fill(mask.unsqueeze(-1), 0.0)

        if dist.is_available() and dist.is_initialized() and self.tp_group is not None:
            out = _all_reduce_autograd(out, self.tp_group)
        return out


class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int,
        tp_rank: int,
        bias: bool = True,
        gather_output: bool = False,
    ):
        super().__init__()
        _check_divisible("out_features", out_features, tp_size)
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group: Optional[dist.ProcessGroup] = None
        self.gather_output = gather_output

        self.local_out = out_features // tp_size
        self.weight = nn.Parameter(torch.empty(self.local_out, in_features))
        nn.init.xavier_uniform_(self.weight)
        _mark_tp_sharded(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.local_out))
            _mark_tp_sharded(self.bias)
        else:
            self.register_parameter("bias", None)

    def set_tp_group(self, group: Optional[dist.ProcessGroup]) -> None:
        self.tp_group = group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if not self.gather_output:
            return out
        return _all_gather_autograd(out, self.tp_group, world_size=self.tp_size)


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int,
        tp_rank: int,
        bias: bool = True,
        input_is_parallel: bool = True,
    ):
        super().__init__()
        _check_divisible("in_features", in_features, tp_size)
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group: Optional[dist.ProcessGroup] = None
        self.input_is_parallel = input_is_parallel

        self.local_in = in_features // tp_size
        self.weight = nn.Parameter(torch.empty(out_features, self.local_in))
        nn.init.xavier_uniform_(self.weight)
        _mark_tp_sharded(self.weight)
        if bias:
            # Bias is replicated across TP ranks.
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def set_tp_group(self, group: Optional[dist.ProcessGroup]) -> None:
        self.tp_group = group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_is_parallel:
            x_local = x
        else:
            start = self.tp_rank * self.local_in
            end = start + self.local_in
            x_local = x[..., start:end]
        out = F.linear(x_local, self.weight, None)
        if self.tp_size > 1 and dist.is_available() and dist.is_initialized() and self.tp_group is not None:
            out = _all_reduce_autograd(out, self.tp_group)
        if self.bias is not None:
            out = out + self.bias
        return out
