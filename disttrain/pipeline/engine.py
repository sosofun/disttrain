from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from disttrain.config import RunConfig
from disttrain.data import FakeBatchProvider
from disttrain.dist.groups import ProcessGroupManager
from disttrain.dist.p2p import (
    P2PRouter,
    activation_tag,
    gradient_tag,
    recv_tensor,
    send_tensor,
)
from disttrain.dist.topology import Topology
from disttrain.models.base import StageModel, TensorDict
from disttrain.pipeline.scheduler import PipelineScheduler, theoretical_bubble_ratio


@dataclass
class StepMetrics:
    step: int
    loss: float
    step_time_sec: float
    forward_time_sec: float
    backward_time_sec: float
    tokens_per_sec: float
    samples_per_sec: float
    bubble_ratio: float
    comm_time_sec: float
    comm_bytes_mb: float
    comm_bandwidth_mb_s: float
    comm_allreduce_sec: float
    comm_allreduce_mb: float
    dataloader_wait_sec: float
    host_to_device_sec: float
    comm_activation_send_sec: float
    comm_activation_recv_sec: float
    comm_gradient_send_sec: float
    comm_gradient_recv_sec: float
    comm_wait_sec: float
    comm_p2p_send_launch_sec: float
    comm_p2p_recv_launch_sec: float
    comm_p2p_recv_wait_sec: float
    comm_p2p_send_wait_sec: float
    comm_p2p_prepost_posted: int
    comm_p2p_prepost_hits: int
    comm_p2p_prepost_misses: int
    comm_p2p_prepost_hit_rate: float
    comm_p2p_recv_overlap_est_sec: float
    comm_p2p_recv_overlap_ratio: float
    gpu_mem_peak_mb: float
    grad_norm: Optional[float]
    lr: float
    scaler_scale: Optional[float]
    sync_impl: str
    optimizer_stepped: bool


class TrainingEngine:
    def __init__(
        self,
        config: RunConfig,
        topology: Topology,
        group_manager: ProcessGroupManager,
        model: StageModel,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler],
        device: torch.device,
    ):
        self.config = config
        self.topology = topology
        self.group_manager = group_manager
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.device = device

        self.local_stage_name = topology.local_stage_name
        self.stage_index = topology.enabled_stage_names.index(self.local_stage_name)
        self.pipeline_depth = len(topology.enabled_stage_names)
        self.is_first_stage = self.stage_index == 0
        self.is_last_stage = self.stage_index == self.pipeline_depth - 1

        self.local_tp_idx = topology.local_tp_index()
        self.local_dp_idx = topology.local_dp_index()
        self.local_stage = topology.local_stage
        self.router = P2PRouter(topology)
        self.use_ddp = isinstance(self.model, DDP)
        # Keep tensor refs together with work handles for async send lifetime.
        self.pending_works: List[Tuple[dist.Work, torch.Tensor]] = []

        precision = config.training.precision.lower()
        self.use_autocast = self.device.type == "cuda" and precision in {"bf16", "fp16"}
        if precision == "bf16":
            self.autocast_dtype = torch.bfloat16
        elif precision == "fp16":
            self.autocast_dtype = torch.float16
        else:
            self.autocast_dtype = torch.float32
        self.transport_dtype = self._resolve_transport_dtype()
        self._direct_prev_boundary = self._resolve_direct_prev_boundary()
        self._direct_next_boundary = self._resolve_direct_next_boundary()

        self.bubble_ratio = theoretical_bubble_ratio(
            pipeline_depth=self.pipeline_depth,
            num_micro_batches=self.config.pipeline.num_micro_batches,
        )
        # 1F1B relies on non-blocking p2p to avoid cross-stage send/recv lockstep deadlocks.
        self._force_async_p2p = self.config.pipeline.schedule == "1f1b"

        self._forward_hidden: Dict[int, torch.Tensor] = {}
        self._forward_inputs: Dict[int, torch.Tensor] = {}
        self._losses: Dict[int, torch.Tensor] = {}
        self._pending_activation_recvs: Dict[int, Tuple[torch.Tensor, dist.Work, float]] = {}
        self._pending_gradient_recvs: Dict[int, Tuple[torch.Tensor, dist.Work, float]] = {}
        self._step_comm_time_sec = 0.0
        self._step_comm_bytes = 0
        self._step_dataloader_wait_sec = 0.0
        self._step_h2d_sec = 0.0
        self._step_comm_breakdown = {
            "act_send": 0.0,
            "act_recv": 0.0,
            "grad_send": 0.0,
            "grad_recv": 0.0,
            "wait": 0.0,
        }
        self._step_p2p_profile = self._new_p2p_profile()
        self.source_provider: Optional[FakeBatchProvider] = None
        if self.is_first_stage:
            self.source_provider = FakeBatchProvider(
                training=self.config.training,
                input_modalities=self.topology.stages[self.local_stage_name].input_modalities,
                device=self.device,
                dp_size=self.local_stage.dp_size,
                dp_rank=self.local_dp_idx,
            )

    def _dist_ready(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    def _resolve_transport_dtype(self) -> torch.dtype:
        raw = self.config.pipeline.transport_dtype
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if raw == "auto":
            if self.device.type == "cuda":
                if self.config.training.precision == "bf16":
                    return torch.bfloat16
                if self.config.training.precision == "fp16":
                    return torch.float16
            return torch.float32

        chosen = dtype_map[raw]
        if self.device.type != "cuda" and chosen != torch.float32:
            if self.topology.runtime_rank == 0:
                print(
                    "[WARN] non-CUDA device does not use low-precision transport, "
                    f"fallback transport dtype {raw} -> fp32."
                )
            return torch.float32
        return chosen

    def _resolve_direct_prev_boundary(self) -> bool:
        prev_stage = self.topology.prev_stage(self.local_stage_name)
        return self._resolve_direct_boundary(peer_stage=prev_stage)

    def _resolve_direct_next_boundary(self) -> bool:
        next_stage = self.topology.next_stage(self.local_stage_name)
        return self._resolve_direct_boundary(peer_stage=next_stage)

    def _resolve_direct_boundary(self, peer_stage: Optional[str]) -> bool:
        if peer_stage is None:
            return False
        mode = self.config.pipeline.transport_tp_mode
        if mode == "single":
            return False
        if self.local_stage.tp_size <= 1:
            return False
        peer_tp = self.topology.stages[peer_stage].tp_size
        if mode == "direct":
            return True
        # auto mode: use direct TP-to-TP boundary only when tp_size matches.
        return peer_tp == self.local_stage.tp_size and peer_tp > 1

    def _transport_rank(self) -> bool:
        # Each TP replica group uses tp_idx=0 as the cross-stage transport rank.
        # Other TP ranks receive via intra-stage TP broadcast.
        return self.local_tp_idx == 0

    def _recv_activation_on_this_rank(self) -> bool:
        return self._direct_prev_boundary or self._transport_rank()

    def _send_activation_on_this_rank(self) -> bool:
        return self._direct_next_boundary or self._transport_rank()

    def _recv_gradient_on_this_rank(self) -> bool:
        return self._direct_next_boundary or self._transport_rank()

    def _send_gradient_on_this_rank(self) -> bool:
        return self._direct_prev_boundary or self._transport_rank()

    def _activation_src_rank(self) -> Optional[int]:
        if self._direct_prev_boundary:
            return self.router.prev_peer_rank_tp(
                self.local_stage_name,
                self.local_dp_idx,
                self.local_tp_idx,
            )
        return self.router.prev_peer_rank(self.local_stage_name, self.local_dp_idx)

    def _activation_dst_rank(self) -> Optional[int]:
        if self._direct_next_boundary:
            return self.router.next_peer_rank_tp(
                self.local_stage_name,
                self.local_dp_idx,
                self.local_tp_idx,
            )
        return self.router.next_peer_rank(self.local_stage_name, self.local_dp_idx)

    def _gradient_src_rank(self) -> Optional[int]:
        if self._direct_next_boundary:
            return self.router.next_peer_rank_tp(
                self.local_stage_name,
                self.local_dp_idx,
                self.local_tp_idx,
            )
        return self.router.next_peer_rank(self.local_stage_name, self.local_dp_idx)

    def _gradient_dst_rank(self) -> Optional[int]:
        if self._direct_prev_boundary:
            return self.router.prev_peer_rank_tp(
                self.local_stage_name,
                self.local_dp_idx,
                self.local_tp_idx,
            )
        return self.router.prev_peer_rank(self.local_stage_name, self.local_dp_idx)

    def _prepost_step_recvs(self, step: int, actions: List) -> None:
        """
        Pre-post irecv for this step's planned receives so comm can progress while
        current micro-batch compute is running. We keep this simple and post one
        receive per micro-batch action.
        """
        if not (self.config.pipeline.overlap_p2p_comm or self._force_async_p2p):
            return

        cfg = self.config.training
        shape = (cfg.micro_batch_size, cfg.seq_len, cfg.hidden_size)
        if not self.is_first_stage and self._recv_activation_on_this_rank():
            src_rank = self._activation_src_rank()
            if src_rank is not None:
                for action in actions:
                    if action.kind != "F":
                        continue
                    mb = action.micro_batch_idx
                    if mb in self._pending_activation_recvs:
                        continue
                    t_launch = time.perf_counter()
                    tensor, work = recv_tensor(
                        shape=shape,
                        dtype=self.transport_dtype,
                        device=self.device,
                        src_rank=src_rank,
                        tag=activation_tag(step, mb),
                        async_op=True,
                    )
                    self._step_p2p_profile["act_recv_launch_sec"] += time.perf_counter() - t_launch
                    if work is not None:
                        self._step_p2p_profile["prepost_posted"] += 1.0
                        self._pending_activation_recvs[mb] = (tensor, work, time.perf_counter())

        if not self.is_last_stage and self._recv_gradient_on_this_rank():
            src_rank = self._gradient_src_rank()
            if src_rank is not None:
                for action in actions:
                    if action.kind != "B":
                        continue
                    mb = action.micro_batch_idx
                    if mb in self._pending_gradient_recvs:
                        continue
                    t_launch = time.perf_counter()
                    tensor, work = recv_tensor(
                        shape=shape,
                        dtype=self.transport_dtype,
                        device=self.device,
                        src_rank=src_rank,
                        tag=gradient_tag(step, mb),
                        async_op=True,
                    )
                    self._step_p2p_profile["grad_recv_launch_sec"] += time.perf_counter() - t_launch
                    if work is not None:
                        self._step_p2p_profile["prepost_posted"] += 1.0
                        self._pending_gradient_recvs[mb] = (tensor, work, time.perf_counter())

    def _estimate_ddp_sync_bytes_mb(self) -> float:
        if not self.use_ddp:
            return 0.0
        if self.local_stage.dp_size <= 1:
            return 0.0
        total_bytes = 0
        for p in self.model.parameters():
            if p.grad is None:
                continue
            total_bytes += p.grad.numel() * p.grad.element_size()
        return float(total_bytes) / (1024.0 * 1024.0)

    def _record_comm(self, channel: str, elapsed_sec: float, tensor: torch.Tensor) -> None:
        self._step_comm_time_sec += elapsed_sec
        self._step_comm_bytes += tensor.numel() * tensor.element_size()
        if channel in self._step_comm_breakdown:
            self._step_comm_breakdown[channel] += elapsed_sec

    def _new_p2p_profile(self) -> Dict[str, float]:
        return {
            "act_send_launch_sec": 0.0,
            "grad_send_launch_sec": 0.0,
            "act_recv_launch_sec": 0.0,
            "grad_recv_launch_sec": 0.0,
            "act_recv_wait_sec": 0.0,
            "grad_recv_wait_sec": 0.0,
            "send_wait_sec": 0.0,
            "prepost_posted": 0.0,
            "act_prepost_hit": 0.0,
            "grad_prepost_hit": 0.0,
            "act_prepost_miss": 0.0,
            "grad_prepost_miss": 0.0,
            # Heuristic overlap: time between irecv post and wait start.
            "recv_overlap_est_sec": 0.0,
        }

    def _tp_broadcast(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self._dist_ready() or self.local_stage.tp_size == 1:
            return tensor
        tp_group = self.group_manager.local_tp_group
        if tp_group is None:
            return tensor
        src_rank = self.topology.rank_for(self.local_stage_name, self.local_dp_idx, 0)
        dist.broadcast(tensor, src=src_rank, group=tp_group)
        return tensor

    def _make_source_inputs(self) -> TensorDict:
        if self.source_provider is not None:
            batch, io_stats = self.source_provider.next_batch()
            self._step_dataloader_wait_sec += float(io_stats.get("dataloader_wait_sec", 0.0))
            self._step_h2d_sec += float(io_stats.get("host_to_device_sec", 0.0))
        else:
            cfg = self.config.training
            batch_size = cfg.micro_batch_size
            seq_len = cfg.seq_len
            batch = {
                "text_tokens": torch.randint(
                    low=0,
                    high=cfg.vocab_size,
                    size=(batch_size, seq_len),
                    device=self.device,
                    dtype=torch.long,
                )
            }

        # Keep TP replicas consistent.
        if self._dist_ready() and self.local_stage.tp_size > 1:
            for k, v in batch.items():
                if not self._transport_rank():
                    batch[k] = torch.empty_like(v)
                self._tp_broadcast(batch[k])
        return batch

    def _recv_activation(self, step: int, micro_batch_idx: int) -> torch.Tensor:
        cfg = self.config.training
        shape = (cfg.micro_batch_size, cfg.seq_len, cfg.hidden_size)
        if self._recv_activation_on_this_rank():
            preposted = self._pending_activation_recvs.pop(micro_batch_idx, None)
            if preposted is not None:
                tensor, work, post_t = preposted
                wait_start = time.perf_counter()
                work.wait()
                waited = time.perf_counter() - wait_start
                self._step_p2p_profile["act_prepost_hit"] += 1.0
                self._step_p2p_profile["act_recv_wait_sec"] += waited
                self._step_p2p_profile["recv_overlap_est_sec"] += max(wait_start - post_t, 0.0)
                self._record_comm("act_recv", waited, tensor)
            else:
                src_rank = self._activation_src_rank()
                if src_rank is None:
                    raise RuntimeError("prev stage is missing for activation recv")
                self._step_p2p_profile["act_prepost_miss"] += 1.0
                t0 = time.perf_counter()
                tensor, _ = recv_tensor(
                    shape=shape,
                    dtype=self.transport_dtype,
                    device=self.device,
                    src_rank=src_rank,
                    tag=activation_tag(step, micro_batch_idx),
                    async_op=False,
                )
                waited = time.perf_counter() - t0
                self._step_p2p_profile["act_recv_wait_sec"] += waited
                self._record_comm("act_recv", waited, tensor)
        else:
            tensor = torch.empty(shape, dtype=self.transport_dtype, device=self.device)
        if not self._direct_prev_boundary:
            self._tp_broadcast(tensor)
        if tensor.dtype != torch.float32:
            tensor = tensor.to(dtype=torch.float32)
        tensor.requires_grad_(True)
        return tensor

    def _send_activation(self, hidden: torch.Tensor, step: int, micro_batch_idx: int) -> None:
        if not self._send_activation_on_this_rank():
            return
        dst_rank = self._activation_dst_rank()
        if dst_rank is None:
            raise RuntimeError("next stage is missing for activation send")
        use_async = self.config.pipeline.overlap_p2p_comm or self._force_async_p2p
        payload = hidden
        if hidden.dtype != self.transport_dtype:
            payload = hidden.to(dtype=self.transport_dtype)
        t0 = time.perf_counter()
        work = send_tensor(
            tensor=payload,
            dst_rank=dst_rank,
            tag=activation_tag(step, micro_batch_idx),
            async_op=use_async,
        )
        launched = time.perf_counter() - t0
        self._step_p2p_profile["act_send_launch_sec"] += launched
        self._record_comm("act_send", launched, payload)
        if work is not None:
            self.pending_works.append(work)

    def _recv_gradient(self, step: int, micro_batch_idx: int) -> torch.Tensor:
        cfg = self.config.training
        shape = (cfg.micro_batch_size, cfg.seq_len, cfg.hidden_size)
        if self._recv_gradient_on_this_rank():
            preposted = self._pending_gradient_recvs.pop(micro_batch_idx, None)
            if preposted is not None:
                grad, work, post_t = preposted
                wait_start = time.perf_counter()
                work.wait()
                waited = time.perf_counter() - wait_start
                self._step_p2p_profile["grad_prepost_hit"] += 1.0
                self._step_p2p_profile["grad_recv_wait_sec"] += waited
                self._step_p2p_profile["recv_overlap_est_sec"] += max(wait_start - post_t, 0.0)
                self._record_comm("grad_recv", waited, grad)
            else:
                src_rank = self._gradient_src_rank()
                if src_rank is None:
                    raise RuntimeError("next stage is missing for gradient recv")
                self._step_p2p_profile["grad_prepost_miss"] += 1.0
                t0 = time.perf_counter()
                grad, _ = recv_tensor(
                    shape=shape,
                    dtype=self.transport_dtype,
                    device=self.device,
                    src_rank=src_rank,
                    tag=gradient_tag(step, micro_batch_idx),
                    async_op=False,
                )
                waited = time.perf_counter() - t0
                self._step_p2p_profile["grad_recv_wait_sec"] += waited
                self._record_comm("grad_recv", waited, grad)
        else:
            grad = torch.empty(shape, dtype=self.transport_dtype, device=self.device)
        if not self._direct_next_boundary:
            self._tp_broadcast(grad)
        if grad.dtype != torch.float32:
            grad = grad.to(dtype=torch.float32)
        return grad

    def _send_gradient(self, grad: torch.Tensor, step: int, micro_batch_idx: int) -> None:
        if self.local_stage.tp_size > 1 and self._dist_ready():
            tp_group = self.group_manager.local_tp_group
            if tp_group is not None:
                dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=tp_group)
                grad /= float(self.local_stage.tp_size)

        if not self._send_gradient_on_this_rank():
            return

        dst_rank = self._gradient_dst_rank()
        if dst_rank is None:
            raise RuntimeError("prev stage is missing for gradient send")
        use_async = self.config.pipeline.overlap_p2p_comm or self._force_async_p2p
        payload = grad
        if grad.dtype != self.transport_dtype:
            payload = grad.to(dtype=self.transport_dtype)
        t0 = time.perf_counter()
        work = send_tensor(
            tensor=payload,
            dst_rank=dst_rank,
            tag=gradient_tag(step, micro_batch_idx),
            async_op=use_async,
        )
        launched = time.perf_counter() - t0
        self._step_p2p_profile["grad_send_launch_sec"] += launched
        self._record_comm("grad_send", launched, payload)
        if work is not None:
            self.pending_works.append(work)

    def _compute_loss(self, outputs: TensorDict) -> torch.Tensor:
        cfg = self.config.training
        weighted_losses: List[Tuple[str, torch.Tensor, float]] = []
        loss_weights = cfg.loss_weights

        if "text_logits" in outputs:
            logits = outputs["text_logits"]
            labels = torch.randint(
                low=0,
                high=cfg.vocab_size,
                size=(logits.size(0), logits.size(1)),
                device=logits.device,
                dtype=torch.long,
            )
            text_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            weighted_losses.append(("text", text_loss, float(loss_weights.get("text", 1.0))))
        elif "logits" in outputs:
            logits = outputs["logits"]
            labels = torch.randint(
                low=0,
                high=cfg.vocab_size,
                size=(logits.size(0), logits.size(1)),
                device=logits.device,
                dtype=torch.long,
            )
            text_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            weighted_losses.append(("text", text_loss, float(loss_weights.get("text", 1.0))))

        if "image_pred" in outputs:
            image_target = torch.zeros_like(outputs["image_pred"])
            image_loss = F.mse_loss(outputs["image_pred"], image_target)
            weighted_losses.append(("image", image_loss, float(loss_weights.get("image", 1.0))))
        if "audio_pred" in outputs:
            audio_target = torch.zeros_like(outputs["audio_pred"])
            audio_loss = F.mse_loss(outputs["audio_pred"], audio_target)
            weighted_losses.append(("audio", audio_loss, float(loss_weights.get("audio", 1.0))))

        if not weighted_losses:
            raise RuntimeError("loss is empty: sink stage produced no supervised outputs")
        active = [(name, loss, w) for name, loss, w in weighted_losses if w > 0]
        if not active:
            raise RuntimeError(
                "all active output losses are disabled by zero weights in training.loss_weights"
            )
        # Normalize by active weight sum so scaling remains stable when tasks are toggled.
        denom = sum(w for _name, _loss, w in active)
        total = sum(loss * w for _name, loss, w in active) / max(float(denom), 1e-12)
        return total / float(self.config.training.grad_accum_steps)

    def _forward_micro_batch(self, step: int, micro_batch_idx: int) -> float:
        if self.is_first_stage:
            inputs = self._make_source_inputs()
        else:
            hidden = self._recv_activation(step=step, micro_batch_idx=micro_batch_idx)
            inputs = {"hidden_states": hidden}
            self._forward_inputs[micro_batch_idx] = hidden

        amp_ctx = (
            torch.autocast(
                device_type=self.device.type,
                dtype=self.autocast_dtype,
                enabled=self.use_autocast,
            )
            if self.device.type in {"cuda", "cpu"}
            else nullcontext()
        )
        with amp_ctx:
            outputs = self.model(inputs, meta={"step": step, "micro_batch_idx": micro_batch_idx})

        if not self.is_last_stage:
            if "hidden_states" not in outputs:
                raise KeyError(
                    f"{self.local_stage_name} forward output must contain 'hidden_states'"
                )
            hidden = outputs["hidden_states"].to(dtype=torch.float32)
            self._forward_hidden[micro_batch_idx] = hidden
            self._send_activation(hidden, step=step, micro_batch_idx=micro_batch_idx)

        if self.is_last_stage:
            loss = self._compute_loss(outputs)
            self._losses[micro_batch_idx] = loss
            return float(loss.detach()) * float(self.config.training.grad_accum_steps)
        return 0.0

    def _backward_micro_batch(self, step: int, micro_batch_idx: int) -> None:
        if self.is_last_stage:
            loss = self._losses[micro_batch_idx]
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            if not self.is_first_stage:
                grad_in = self._forward_inputs[micro_batch_idx].grad
                if grad_in is None:
                    raise RuntimeError("missing input grad at sink stage")
                if self.scaler is not None:
                    scale = float(self.scaler.get_scale())
                    grad_to_send = grad_in / max(scale, 1.0)
                else:
                    grad_to_send = grad_in
                self._send_gradient(grad_to_send, step=step, micro_batch_idx=micro_batch_idx)
            return

        grad_out = self._recv_gradient(step=step, micro_batch_idx=micro_batch_idx)
        hidden = self._forward_hidden[micro_batch_idx]
        hidden.backward(grad_out)
        if not self.is_first_stage:
            grad_in = self._forward_inputs[micro_batch_idx].grad
            if grad_in is None:
                raise RuntimeError("missing input grad at middle stage")
            self._send_gradient(grad_in, step=step, micro_batch_idx=micro_batch_idx)

    def _wait_pending(self) -> None:
        t0 = time.perf_counter()
        for work, _payload in self.pending_works:
            work.wait()
        if self.pending_works:
            waited = time.perf_counter() - t0
            self._step_comm_time_sec += waited
            self._step_comm_breakdown["wait"] += waited
            self._step_p2p_profile["send_wait_sec"] += waited
        self.pending_works.clear()

    def _run_pipeline_step(self, step: int) -> Dict[str, float]:
        self._forward_hidden.clear()
        self._forward_inputs.clear()
        self._losses.clear()
        self._pending_activation_recvs.clear()
        self._pending_gradient_recvs.clear()
        self._step_comm_time_sec = 0.0
        self._step_comm_bytes = 0
        self._step_dataloader_wait_sec = 0.0
        self._step_h2d_sec = 0.0
        self._step_comm_breakdown = {
            "act_send": 0.0,
            "act_recv": 0.0,
            "grad_send": 0.0,
            "grad_recv": 0.0,
            "wait": 0.0,
        }
        self._step_p2p_profile = self._new_p2p_profile()

        scheduler = PipelineScheduler(
            schedule=self.config.pipeline.schedule,
            num_micro_batches=self.config.pipeline.num_micro_batches,
            pipeline_depth=self.pipeline_depth,
            stage_index=self.stage_index,
        )
        actions = scheduler.build()
        self._prepost_step_recvs(step, actions)

        total_loss = 0.0
        forward_time = 0.0
        backward_time = 0.0
        for action in actions:
            if action.kind == "F":
                t0 = time.perf_counter()
                total_loss += self._forward_micro_batch(step, action.micro_batch_idx)
                forward_time += time.perf_counter() - t0
            else:
                t0 = time.perf_counter()
                self._backward_micro_batch(step, action.micro_batch_idx)
                backward_time += time.perf_counter() - t0

        self._wait_pending()
        self._pending_activation_recvs.clear()
        self._pending_gradient_recvs.clear()
        comm_mb = float(self._step_comm_bytes) / (1024.0 * 1024.0)
        comm_time = float(self._step_comm_time_sec)
        comm_bw = comm_mb / max(comm_time, 1e-6)
        p2p_send_launch = float(
            self._step_p2p_profile["act_send_launch_sec"]
            + self._step_p2p_profile["grad_send_launch_sec"]
        )
        p2p_recv_launch = float(
            self._step_p2p_profile["act_recv_launch_sec"]
            + self._step_p2p_profile["grad_recv_launch_sec"]
        )
        p2p_recv_wait = float(
            self._step_p2p_profile["act_recv_wait_sec"]
            + self._step_p2p_profile["grad_recv_wait_sec"]
        )
        p2p_send_wait = float(self._step_p2p_profile["send_wait_sec"])
        p2p_prepost_posted = int(self._step_p2p_profile["prepost_posted"])
        p2p_prepost_hits = int(
            self._step_p2p_profile["act_prepost_hit"] + self._step_p2p_profile["grad_prepost_hit"]
        )
        p2p_prepost_misses = int(
            self._step_p2p_profile["act_prepost_miss"] + self._step_p2p_profile["grad_prepost_miss"]
        )
        prepost_total = p2p_prepost_hits + p2p_prepost_misses
        p2p_prepost_hit_rate = (
            float(p2p_prepost_hits) / float(prepost_total) if prepost_total > 0 else 0.0
        )
        p2p_recv_overlap_est = float(self._step_p2p_profile["recv_overlap_est_sec"])
        p2p_recv_overlap_ratio = p2p_recv_overlap_est / max(
            p2p_recv_overlap_est + p2p_recv_wait,
            1e-12,
        )
        return {
            "loss": total_loss / max(self.config.pipeline.num_micro_batches, 1),
            "forward_time": forward_time,
            "backward_time": backward_time,
            "comm_time": comm_time,
            "comm_mb": comm_mb,
            "comm_bw": comm_bw,
            "comm_act_send": self._step_comm_breakdown["act_send"],
            "comm_act_recv": self._step_comm_breakdown["act_recv"],
            "comm_grad_send": self._step_comm_breakdown["grad_send"],
            "comm_grad_recv": self._step_comm_breakdown["grad_recv"],
            "comm_wait": self._step_comm_breakdown["wait"],
            "p2p_send_launch": p2p_send_launch,
            "p2p_recv_launch": p2p_recv_launch,
            "p2p_recv_wait": p2p_recv_wait,
            "p2p_send_wait": p2p_send_wait,
            "p2p_prepost_posted": p2p_prepost_posted,
            "p2p_prepost_hits": p2p_prepost_hits,
            "p2p_prepost_misses": p2p_prepost_misses,
            "p2p_prepost_hit_rate": p2p_prepost_hit_rate,
            "p2p_recv_overlap_est": p2p_recv_overlap_est,
            "p2p_recv_overlap_ratio": p2p_recv_overlap_ratio,
            "dataloader_wait": self._step_dataloader_wait_sec,
            "host_to_device": self._step_h2d_sec,
        }

    def run(self, max_steps: int) -> List[StepMetrics]:
        self.model.train()
        metrics: List[StepMetrics] = []

        grad_accum_steps = self.config.training.grad_accum_steps
        if grad_accum_steps < 1:
            grad_accum_steps = 1

        for step in range(max_steps):
            if step % grad_accum_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)

            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            should_step = (step + 1) % grad_accum_steps == 0
            t0 = time.perf_counter()
            if self.use_ddp and not should_step:
                with self.model.no_sync():
                    out = self._run_pipeline_step(step)
            else:
                out = self._run_pipeline_step(step)
            grad_norm_value: Optional[float] = None
            scaler_scale: Optional[float] = (
                float(self.scaler.get_scale()) if self.scaler is not None else None
            )
            sync_stats = {"time_sec": 0.0, "bytes_mb": 0.0}
            opt_sync_stats = {"time_sec": 0.0, "bytes_mb": 0.0}
            sync_impl = "none"
            if should_step:
                zero_stage = int(getattr(self.optimizer, "zero_stage", 0))
                if self.scaler is not None:
                    # Clip/sync should see unscaled gradients.
                    self.scaler.unscale_(self.optimizer)
                if self.use_ddp:
                    # DP gradients are synchronized by DDP hooks.
                    sync_impl = "ddp"
                    sync_stats["bytes_mb"] += self._estimate_ddp_sync_bytes_mb()
                    # Keep TP synchronization explicit when tp_size > 1.
                    tp_stats = self.group_manager.average_gradients(
                        self.model,
                        bucket_mb=self.config.distributed.grad_sync_bucket_mb,
                        sync_tp=True,
                        sync_dp=False,
                    )
                    sync_stats["time_sec"] += tp_stats["time_sec"]
                    sync_stats["bytes_mb"] += tp_stats["bytes_mb"]
                    if self.local_stage.tp_size > 1:
                        sync_impl = "ddp+tp_manual"
                else:
                    if zero_stage == 1:
                        # Distributed optimizer handles DP sync via reduce-scatter/all-gather.
                        sync_impl = "zero1"
                        if self.local_stage.tp_size > 1:
                            # ZeRO-1 replaces DP sync, but TP-replicated grads still need averaging.
                            tp_stats = self.group_manager.average_gradients(
                                self.model,
                                bucket_mb=self.config.distributed.grad_sync_bucket_mb,
                                sync_tp=True,
                                sync_dp=False,
                            )
                            sync_stats["time_sec"] += tp_stats["time_sec"]
                            sync_stats["bytes_mb"] += tp_stats["bytes_mb"]
                            sync_impl = "tp_manual+zero1"
                    else:
                        sync_impl = "manual"
                        sync_stats = self.group_manager.average_gradients(
                            self.model,
                            bucket_mb=self.config.distributed.grad_sync_bucket_mb,
                            sync_tp=True,
                            sync_dp=True,
                        )
                if self.config.training.grad_clip_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.training.grad_clip_norm,
                    )
                    grad_norm_value = float(grad_norm)
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    scaler_scale = float(self.scaler.get_scale())
                else:
                    self.optimizer.step()
                if zero_stage == 1:
                    # Include optimizer-internal RS/AG traffic in communication accounting.
                    if sync_impl == "none":
                        sync_impl = "zero1"
                    elif "zero1" not in sync_impl:
                        sync_impl = f"{sync_impl}+zero1"
                    zero_stats = getattr(self.optimizer, "last_sync_stats", None)
                    if isinstance(zero_stats, dict):
                        opt_sync_stats["time_sec"] = float(zero_stats.get("time_sec", 0.0))
                        opt_sync_stats["bytes_mb"] = float(zero_stats.get("bytes_mb", 0.0))

            step_time = time.perf_counter() - t0
            total_comm_time = (
                float(out["comm_time"])
                + float(sync_stats["time_sec"])
                + float(opt_sync_stats["time_sec"])
            )
            total_comm_mb = (
                float(out["comm_mb"])
                + float(sync_stats["bytes_mb"])
                + float(opt_sync_stats["bytes_mb"])
            )
            total_comm_bw = total_comm_mb / max(total_comm_time, 1e-6)
            tokens = (
                self.config.training.micro_batch_size
                * self.config.training.seq_len
                * self.config.pipeline.num_micro_batches
            )
            samples = (
                self.config.training.micro_batch_size
                * self.config.pipeline.num_micro_batches
            )
            tokens_per_sec = float(tokens) / max(step_time, 1e-6)
            samples_per_sec = float(samples) / max(step_time, 1e-6)
            gpu_mem_peak_mb = 0.0
            if self.device.type == "cuda":
                gpu_mem_peak_mb = float(torch.cuda.max_memory_allocated(self.device)) / (
                    1024.0 * 1024.0
                )
            metrics.append(
                StepMetrics(
                    step=step,
                    loss=float(out["loss"]),
                    step_time_sec=step_time,
                    forward_time_sec=float(out["forward_time"]),
                    backward_time_sec=float(out["backward_time"]),
                    tokens_per_sec=tokens_per_sec,
                    samples_per_sec=samples_per_sec,
                    bubble_ratio=self.bubble_ratio,
                    comm_time_sec=total_comm_time,
                    comm_bytes_mb=total_comm_mb,
                    comm_bandwidth_mb_s=total_comm_bw,
                    comm_allreduce_sec=float(sync_stats["time_sec"]),
                    comm_allreduce_mb=float(sync_stats["bytes_mb"]),
                    dataloader_wait_sec=float(out["dataloader_wait"]),
                    host_to_device_sec=float(out["host_to_device"]),
                    comm_activation_send_sec=float(out["comm_act_send"]),
                    comm_activation_recv_sec=float(out["comm_act_recv"]),
                    comm_gradient_send_sec=float(out["comm_grad_send"]),
                    comm_gradient_recv_sec=float(out["comm_grad_recv"]),
                    comm_wait_sec=float(out["comm_wait"]),
                    comm_p2p_send_launch_sec=float(out["p2p_send_launch"]),
                    comm_p2p_recv_launch_sec=float(out["p2p_recv_launch"]),
                    comm_p2p_recv_wait_sec=float(out["p2p_recv_wait"]),
                    comm_p2p_send_wait_sec=float(out["p2p_send_wait"]),
                    comm_p2p_prepost_posted=int(out["p2p_prepost_posted"]),
                    comm_p2p_prepost_hits=int(out["p2p_prepost_hits"]),
                    comm_p2p_prepost_misses=int(out["p2p_prepost_misses"]),
                    comm_p2p_prepost_hit_rate=float(out["p2p_prepost_hit_rate"]),
                    comm_p2p_recv_overlap_est_sec=float(out["p2p_recv_overlap_est"]),
                    comm_p2p_recv_overlap_ratio=float(out["p2p_recv_overlap_ratio"]),
                    gpu_mem_peak_mb=gpu_mem_peak_mb,
                    grad_norm=grad_norm_value,
                    lr=float(self.optimizer.param_groups[0].get("lr", 0.0)),
                    scaler_scale=scaler_scale,
                    sync_impl=sync_impl,
                    optimizer_stepped=should_step,
                )
            )
        return metrics
