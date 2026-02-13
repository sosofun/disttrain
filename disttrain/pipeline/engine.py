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
    comm_activation_send_sec: float
    comm_activation_recv_sec: float
    comm_gradient_send_sec: float
    comm_gradient_recv_sec: float
    comm_wait_sec: float
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

        self.bubble_ratio = theoretical_bubble_ratio(
            pipeline_depth=self.pipeline_depth,
            num_micro_batches=self.config.pipeline.num_micro_batches,
        )
        # 1F1B relies on non-blocking p2p to avoid cross-stage send/recv lockstep deadlocks.
        self._force_async_p2p = self.config.pipeline.schedule == "1f1b"

        self._forward_hidden: Dict[int, torch.Tensor] = {}
        self._forward_inputs: Dict[int, torch.Tensor] = {}
        self._losses: Dict[int, torch.Tensor] = {}
        self._step_comm_time_sec = 0.0
        self._step_comm_bytes = 0
        self._step_comm_breakdown = {
            "act_send": 0.0,
            "act_recv": 0.0,
            "grad_send": 0.0,
            "grad_recv": 0.0,
            "wait": 0.0,
        }

    def _dist_ready(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    def _transport_rank(self) -> bool:
        return self.local_tp_idx == 0

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
        cfg = self.config.training
        batch_size = cfg.micro_batch_size
        seq_len = cfg.seq_len

        batch: TensorDict = {
            "text_tokens": torch.randint(
                low=0,
                high=cfg.vocab_size,
                size=(batch_size, seq_len),
                device=self.device,
                dtype=torch.long,
            )
        }

        input_modalities = self.topology.stages[self.local_stage_name].input_modalities
        if "image" in input_modalities:
            batch["image"] = torch.randn(
                batch_size,
                3,
                cfg.image_size,
                cfg.image_size,
                device=self.device,
            )
        if "video" in input_modalities:
            batch["video"] = torch.randn(
                batch_size,
                cfg.video_frames,
                3,
                cfg.image_size,
                cfg.image_size,
                device=self.device,
            )
        if "audio" in input_modalities:
            batch["audio"] = torch.randn(
                batch_size,
                1,
                cfg.audio_length,
                device=self.device,
            )

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
        if self._transport_rank():
            src_rank = self.router.prev_peer_rank(self.local_stage_name, self.local_dp_idx)
            if src_rank is None:
                raise RuntimeError("prev stage is missing for activation recv")
            t0 = time.perf_counter()
            tensor, _ = recv_tensor(
                shape=shape,
                dtype=torch.float32,
                device=self.device,
                src_rank=src_rank,
                tag=activation_tag(step, micro_batch_idx),
                async_op=False,
            )
            self._record_comm("act_recv", time.perf_counter() - t0, tensor)
        else:
            tensor = torch.empty(shape, dtype=torch.float32, device=self.device)
        self._tp_broadcast(tensor)
        tensor.requires_grad_(True)
        return tensor

    def _send_activation(self, hidden: torch.Tensor, step: int, micro_batch_idx: int) -> None:
        if not self._transport_rank():
            return
        dst_rank = self.router.next_peer_rank(self.local_stage_name, self.local_dp_idx)
        if dst_rank is None:
            raise RuntimeError("next stage is missing for activation send")
        use_async = self.config.pipeline.overlap_p2p_comm or self._force_async_p2p
        t0 = time.perf_counter()
        work = send_tensor(
            tensor=hidden,
            dst_rank=dst_rank,
            tag=activation_tag(step, micro_batch_idx),
            async_op=use_async,
        )
        self._record_comm("act_send", time.perf_counter() - t0, hidden)
        if work is not None:
            self.pending_works.append(work)

    def _recv_gradient(self, step: int, micro_batch_idx: int) -> torch.Tensor:
        cfg = self.config.training
        shape = (cfg.micro_batch_size, cfg.seq_len, cfg.hidden_size)
        if self._transport_rank():
            src_rank = self.router.next_peer_rank(self.local_stage_name, self.local_dp_idx)
            if src_rank is None:
                raise RuntimeError("next stage is missing for gradient recv")
            t0 = time.perf_counter()
            grad, _ = recv_tensor(
                shape=shape,
                dtype=torch.float32,
                device=self.device,
                src_rank=src_rank,
                tag=gradient_tag(step, micro_batch_idx),
                async_op=False,
            )
            self._record_comm("grad_recv", time.perf_counter() - t0, grad)
        else:
            grad = torch.empty(shape, dtype=torch.float32, device=self.device)
        self._tp_broadcast(grad)
        return grad

    def _send_gradient(self, grad: torch.Tensor, step: int, micro_batch_idx: int) -> None:
        if self.local_stage.tp_size > 1 and self._dist_ready():
            tp_group = self.group_manager.local_tp_group
            if tp_group is not None:
                dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=tp_group)
                grad /= float(self.local_stage.tp_size)

        if not self._transport_rank():
            return

        dst_rank = self.router.prev_peer_rank(self.local_stage_name, self.local_dp_idx)
        if dst_rank is None:
            raise RuntimeError("prev stage is missing for gradient send")
        use_async = self.config.pipeline.overlap_p2p_comm or self._force_async_p2p
        t0 = time.perf_counter()
        work = send_tensor(
            tensor=grad,
            dst_rank=dst_rank,
            tag=gradient_tag(step, micro_batch_idx),
            async_op=use_async,
        )
        self._record_comm("grad_send", time.perf_counter() - t0, grad)
        if work is not None:
            self.pending_works.append(work)

    def _compute_loss(self, outputs: TensorDict) -> torch.Tensor:
        cfg = self.config.training
        losses: List[torch.Tensor] = []

        if "text_logits" in outputs:
            logits = outputs["text_logits"]
            labels = torch.randint(
                low=0,
                high=cfg.vocab_size,
                size=(logits.size(0), logits.size(1)),
                device=logits.device,
                dtype=torch.long,
            )
            losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1)))
        elif "logits" in outputs:
            logits = outputs["logits"]
            labels = torch.randint(
                low=0,
                high=cfg.vocab_size,
                size=(logits.size(0), logits.size(1)),
                device=logits.device,
                dtype=torch.long,
            )
            losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1)))

        if "image_pred" in outputs:
            image_target = torch.zeros_like(outputs["image_pred"])
            losses.append(F.mse_loss(outputs["image_pred"], image_target))
        if "audio_pred" in outputs:
            audio_target = torch.zeros_like(outputs["audio_pred"])
            losses.append(F.mse_loss(outputs["audio_pred"], audio_target))

        if not losses:
            raise RuntimeError("loss is empty: sink stage produced no supervised outputs")

        total = sum(losses) / float(len(losses))
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
        self.pending_works.clear()

    def _run_pipeline_step(self, step: int) -> Dict[str, float]:
        self._forward_hidden.clear()
        self._forward_inputs.clear()
        self._losses.clear()
        self._step_comm_time_sec = 0.0
        self._step_comm_bytes = 0
        self._step_comm_breakdown = {
            "act_send": 0.0,
            "act_recv": 0.0,
            "grad_send": 0.0,
            "grad_recv": 0.0,
            "wait": 0.0,
        }

        scheduler = PipelineScheduler(
            schedule=self.config.pipeline.schedule,
            num_micro_batches=self.config.pipeline.num_micro_batches,
            pipeline_depth=self.pipeline_depth,
            stage_index=self.stage_index,
        )
        actions = scheduler.build()

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
        comm_mb = float(self._step_comm_bytes) / (1024.0 * 1024.0)
        comm_time = float(self._step_comm_time_sec)
        comm_bw = comm_mb / max(comm_time, 1e-6)
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
            sync_impl = "none"
            if should_step:
                if self.scaler is not None:
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

            step_time = time.perf_counter() - t0
            total_comm_time = float(out["comm_time"]) + float(sync_stats["time_sec"])
            total_comm_mb = float(out["comm_mb"]) + float(sync_stats["bytes_mb"])
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
                    comm_activation_send_sec=float(out["comm_act_send"]),
                    comm_activation_recv_sec=float(out["comm_act_recv"]),
                    comm_gradient_send_sec=float(out["comm_grad_send"]),
                    comm_gradient_recv_sec=float(out["comm_grad_recv"]),
                    comm_wait_sec=float(out["comm_wait"]),
                    gpu_mem_peak_mb=gpu_mem_peak_mb,
                    grad_norm=grad_norm_value,
                    lr=float(self.optimizer.param_groups[0].get("lr", 0.0)),
                    scaler_scale=scaler_scale,
                    sync_impl=sync_impl,
                    optimizer_stepped=should_step,
                )
            )
        return metrics
