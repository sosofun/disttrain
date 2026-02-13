from __future__ import annotations

import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from disttrain.models.sequence_parallel import (
    sequence_all_gather,
    sequence_reduce_scatter,
)
from disttrain.models.tp_layers import ColumnParallelLinear, RowParallelLinear


def _require_divisible(name: str, value: int, divisor: int) -> None:
    if value % divisor != 0:
        raise ValueError(f"{name}={value} must be divisible by {divisor}")


class TPSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        tp_size: int,
        tp_rank: int,
        causal: bool = False,
    ):
        super().__init__()
        _require_divisible("hidden_size", hidden_size, num_heads)
        _require_divisible("num_heads", num_heads, tp_size)
        _require_divisible("hidden_size", hidden_size, tp_size)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.causal = causal

        self.head_dim = hidden_size // num_heads
        self.local_heads = num_heads // tp_size
        self.local_hidden = hidden_size // tp_size
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv_proj = ColumnParallelLinear(
            in_features=hidden_size,
            out_features=3 * hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            gather_output=False,
        )
        self.out_proj = RowParallelLinear(
            in_features=hidden_size,
            out_features=hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            input_is_parallel=True,
        )

    def set_tp_group(self, tp_group: Optional[object]) -> None:
        self.qkv_proj.set_tp_group(tp_group)  # type: ignore[arg-type]
        self.out_proj.set_tp_group(tp_group)  # type: ignore[arg-type]

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = hidden.shape
        qkv = self.qkv_proj(hidden)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(batch, seq, self.local_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.local_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.local_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.causal:
            mask = torch.triu(
                torch.ones(seq, seq, device=scores.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(mask, float("-inf"))

        probs = F.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v)
        ctx = ctx.transpose(1, 2).contiguous().view(batch, seq, self.local_hidden)
        out = self.out_proj(ctx)
        return out


class TPFeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        tp_size: int,
        tp_rank: int,
        ffn_hidden_size: Optional[int] = None,
    ):
        super().__init__()
        ffn_hidden = ffn_hidden_size if ffn_hidden_size is not None else hidden_size * 4
        self.fc1 = ColumnParallelLinear(
            in_features=hidden_size,
            out_features=ffn_hidden,
            tp_size=tp_size,
            tp_rank=tp_rank,
            gather_output=False,
        )
        self.act = nn.GELU()
        self.fc2 = RowParallelLinear(
            in_features=ffn_hidden,
            out_features=hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            input_is_parallel=True,
        )

    def set_tp_group(self, tp_group: Optional[object]) -> None:
        self.fc1.set_tp_group(tp_group)  # type: ignore[arg-type]
        self.fc2.set_tp_group(tp_group)  # type: ignore[arg-type]

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(hidden)
        hidden = self.act(hidden)
        hidden = self.fc2(hidden)
        return hidden


class TPTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        tp_size: int,
        tp_rank: int,
        causal: bool = False,
        sequence_parallel: bool = False,
    ):
        super().__init__()
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.sequence_parallel = sequence_parallel and tp_size > 1
        self.tp_group: Optional[dist.ProcessGroup] = None
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = TPSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            tp_size=tp_size,
            tp_rank=tp_rank,
            causal=causal,
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = TPFeedForward(
            hidden_size=hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )

    def set_tp_group(self, tp_group: Optional[object]) -> None:
        self.tp_group = tp_group  # type: ignore[assignment]
        self.attn.set_tp_group(tp_group)
        self.ffn.set_tp_group(tp_group)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.sequence_parallel:
            hidden = hidden + self.attn(self.norm1(hidden))
            hidden = hidden + self.ffn(self.norm2(hidden))
            return hidden

        # Sequence parallel:
        # LN/dropout-like ops run on local sequence shard; TP core runs on full
        # sequence between all-gather/reduce-scatter conjugate ops.
        local = hidden
        x = self.norm1(local)
        x = sequence_all_gather(x, self.tp_group, self.tp_size)
        x = self.attn(x)
        x = sequence_reduce_scatter(x, self.tp_group, self.tp_size)
        local = local + x

        y = self.norm2(local)
        y = sequence_all_gather(y, self.tp_group, self.tp_size)
        y = self.ffn(y)
        y = sequence_reduce_scatter(y, self.tp_group, self.tp_size)
        local = local + y
        return local
