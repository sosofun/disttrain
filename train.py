from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import random
import os
import json
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from disttrain.checkpoint import load_checkpoint, save_checkpoint
from disttrain.config import ConfigError, RunConfig, load_config
from disttrain.dist.groups import ProcessGroupManager
from disttrain.dist.topology import Topology, TopologyError
from disttrain.models.registry import build_stage_model
from disttrain.optim import Zero1AdamW
from disttrain.pipeline.engine import TrainingEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="disttrain entrypoint")
    parser.add_argument("--config", required=True, help="path to yaml/json config")
    parser.add_argument("--max-steps", type=int, default=None, help="override max_steps")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--resume", type=str, default="", help="checkpoint path to resume")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="",
        help="directory to save final checkpoint",
    )
    parser.add_argument(
        "--log-format",
        choices=("text", "json"),
        default="text",
        help="training log output format",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="",
        help="optional file path to append structured json logs",
    )
    parser.add_argument(
        "--no-restore-rng",
        action="store_true",
        help="do not restore RNG state when resuming from checkpoint",
    )
    parser.add_argument(
        "--log-all-ranks",
        action="store_true",
        help="emit metrics logs from all ranks (for benchmark/profiling)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="force deterministic algorithms and backend behavior",
    )
    return parser.parse_args()


def dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_seed(seed: int, rank: int) -> None:
    final_seed = seed + rank
    random.seed(final_seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(final_seed % (2**32))
    except Exception:
        # NumPy is optional for this repo; skip when unavailable.
        pass
    torch.manual_seed(final_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(final_seed)


def configure_determinism(enabled: bool) -> None:
    if not enabled:
        return
    # Required by some CUDA deterministic GEMM kernels.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def init_distributed(config: RunConfig) -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    backend = config.distributed.backend
    # Bind CUDA device before NCCL PG init to avoid "device currently unknown" warnings
    # and potential hangs caused by ambiguous rank->device mapping.
    if world_size > 1 and backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if world_size > 1 and not dist_ready():
        if backend == "nccl" and not torch.cuda.is_available():
            print("[WARN] NCCL requested but CUDA unavailable, fallback to gloo.")
            backend = "gloo"
        timeout = timedelta(seconds=config.distributed.timeout_sec)
        dist.init_process_group(
            backend=backend,
            init_method=config.distributed.init_method,
            timeout=timeout,
        )
        world_size = dist.get_world_size()
        rank = dist.get_rank()

    requested_device = config.training.device
    backend = config.distributed.backend

    if requested_device == "cpu":
        device = torch.device("cpu")
    elif requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("training.device is 'cuda' but CUDA is unavailable")
        if backend == "gloo":
            print("[WARN] backend=gloo with training.device=cuda is unsupported for p2p, fallback to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
    else:  # auto
        if torch.cuda.is_available() and backend != "gloo":
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            if torch.cuda.is_available() and backend == "gloo" and rank == 0:
                print("[WARN] backend=gloo detected, force CPU device for safe tensor transport.")
            device = torch.device("cpu")

    return world_size, rank, local_rank, device


def is_log_rank(topology: Topology) -> bool:
    if topology.runtime_rank == 0:
        return True
    return (
        topology.local_stage_name == topology.enabled_stage_names[-1]
        and topology.local_tp_index() == 0
        and topology.local_dp_index() == 0
    )


def should_log_rank(args: argparse.Namespace, topology: Topology) -> bool:
    if args.log_all_ranks:
        return True
    return is_log_rank(topology)


def rank_log_file_path(base_path: str, rank: int, log_all_ranks: bool) -> Path:
    path = Path(base_path)
    if not log_all_ranks:
        return path
    if path.suffix:
        return path.with_name(f"{path.stem}.rank{rank}{path.suffix}")
    return Path(str(path) + f".rank{rank}")


def main() -> int:
    args = parse_args()

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"[ERROR] config error: {exc}")
        return 2

    if args.deterministic:
        cfg.training.deterministic = True
    configure_determinism(cfg.training.deterministic)

    world_size, rank, _, device = init_distributed(cfg)
    setup_seed(args.seed, rank)
    if rank == 0 and cfg.training.deterministic:
        print("[INFO] deterministic mode enabled (torch/cudnn/cublas/tf32 configured)")

    try:
        topology = Topology(cfg, runtime_world_size=world_size, runtime_rank=rank)
    except TopologyError as exc:
        print(f"[ERROR][rank={rank}] topology error: {exc}")
        if dist_ready():
            dist.destroy_process_group()
        return 3

    if rank == 0:
        for line in topology.summary_lines():
            print(line)

    stage_cfg = cfg.stages[topology.local_stage_name]
    base_model = build_stage_model(
        stage_cfg,
        cfg.training,
        tp_size=topology.local_stage.tp_size,
        tp_rank=topology.local_tp_index(),
    ).to(device)
    stage_lr = cfg.training.optimizer.stage_lrs.get(
        topology.local_stage_name,
        cfg.training.optimizer.lr,
    )
    group_manager = ProcessGroupManager(topology)
    group_manager.create()
    _bind_tp_group_to_model(base_model, group_manager.local_tp_group)
    group_manager.sync_parameters(base_model)
    model = _wrap_model_with_ddp_if_needed(
        model=base_model,
        cfg=cfg,
        topology=topology,
        group_manager=group_manager,
        device=device,
    )

    optimizer = _build_optimizer(
        cfg=cfg,
        topology=topology,
        group_manager=group_manager,
        params=base_model.parameters(),
        lr=stage_lr,
    )
    scaler = _build_scaler_if_needed(cfg, topology, device)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(
            args.resume,
            base_model,
            optimizer,
            scaler,
            topology,
            restore_rng=not args.no_restore_rng,
        )
        if should_log_rank(args, topology):
            print(f"[INFO][rank={rank}] resumed from {args.resume}, step={start_step}")

    engine = TrainingEngine(
        config=cfg,
        topology=topology,
        group_manager=group_manager,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
    )

    max_steps = args.max_steps if args.max_steps is not None else cfg.training.max_steps
    if max_steps <= start_step:
        max_steps = start_step + 1

    log_fp = None
    if args.log_file and should_log_rank(args, topology):
        log_path = rank_log_file_path(args.log_file, rank, args.log_all_ranks)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = log_path.open("a", encoding="utf-8")

    metrics = engine.run(max_steps=max_steps - start_step)
    if should_log_rank(args, topology):
        sp_enabled = bool(
            cfg.stages[topology.local_stage_name].sequence_parallel
            and topology.local_stage.tp_size > 1
        )
        for m in metrics:
            payload = {
                "step": start_step + m.step,
                "rank": rank,
                "stage": topology.local_stage_name,
                "local_tp_idx": topology.local_tp_index(),
                "local_dp_idx": topology.local_dp_index(),
                "sequence_parallel": sp_enabled,
                "loss": m.loss,
                "step_time_sec": m.step_time_sec,
                "forward_time_sec": m.forward_time_sec,
                "backward_time_sec": m.backward_time_sec,
                "tokens_per_sec": m.tokens_per_sec,
                "samples_per_sec": m.samples_per_sec,
                "bubble_ratio": m.bubble_ratio,
                "comm_time_sec": m.comm_time_sec,
                "comm_bytes_mb": m.comm_bytes_mb,
                "comm_bandwidth_mb_s": m.comm_bandwidth_mb_s,
                "comm_allreduce_sec": m.comm_allreduce_sec,
                "comm_allreduce_mb": m.comm_allreduce_mb,
                "dataloader_wait_sec": m.dataloader_wait_sec,
                "host_to_device_sec": m.host_to_device_sec,
                "comm_activation_send_sec": m.comm_activation_send_sec,
                "comm_activation_recv_sec": m.comm_activation_recv_sec,
                "comm_gradient_send_sec": m.comm_gradient_send_sec,
                "comm_gradient_recv_sec": m.comm_gradient_recv_sec,
                "comm_wait_sec": m.comm_wait_sec,
                "gpu_mem_peak_mb": m.gpu_mem_peak_mb,
                "grad_norm": m.grad_norm,
                "lr": m.lr,
                "scaler_scale": m.scaler_scale,
                "sync_impl": m.sync_impl,
                "optimizer_step": m.optimizer_stepped,
            }
            if args.log_format == "json":
                print(json.dumps(payload, ensure_ascii=False))
            else:
                print(
                    "[step={:04d}] loss={:.6f} step_time={:.3f}s fwd={:.3f}s bwd={:.3f}s "
                    "tokens/s={:.1f} samples/s={:.1f} comm={:.3f}s "
                    "io(wait={:.3f},h2d={:.3f}) "
                    "(allr={:.3f},act_s={:.3f},act_r={:.3f},grad_s={:.3f},grad_r={:.3f},wait={:.3f}) "
                    "bw={:.2f}MB/s grad_norm={} lr={:.6g} scaler={} sync={} bubble={:.4f} optimizer_step={}".format(
                        payload["step"],
                        m.loss,
                        m.step_time_sec,
                        m.forward_time_sec,
                        m.backward_time_sec,
                        m.tokens_per_sec,
                        m.samples_per_sec,
                        m.comm_time_sec,
                        m.dataloader_wait_sec,
                        m.host_to_device_sec,
                        m.comm_allreduce_sec,
                        m.comm_activation_send_sec,
                        m.comm_activation_recv_sec,
                        m.comm_gradient_send_sec,
                        m.comm_gradient_recv_sec,
                        m.comm_wait_sec,
                        m.comm_bandwidth_mb_s,
                        "n/a" if m.grad_norm is None else f"{m.grad_norm:.4f}",
                        m.lr,
                        "n/a" if m.scaler_scale is None else f"{m.scaler_scale:.1f}",
                        m.sync_impl,
                        m.bubble_ratio,
                        m.optimizer_stepped,
                    )
                )
            if log_fp is not None:
                log_fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

    if log_fp is not None:
        log_fp.close()

    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / (
            f"stage_{topology.local_stage_name}_rank_{topology.runtime_rank}.pt"
        )
        save_checkpoint(
            path=str(ckpt_path),
            model=base_model,
            optimizer=optimizer,
            scaler=scaler,
            step=start_step + len(metrics),
            config=cfg,
            topology=topology,
        )
        if should_log_rank(args, topology):
            print(f"[INFO][rank={rank}] checkpoint saved: {ckpt_path}")

    group_manager.clear()
    if dist_ready():
        dist.barrier()
        dist.destroy_process_group()
    return 0


def _build_scaler_if_needed(
    cfg: RunConfig,
    topology: Topology,
    device: torch.device,
) -> Optional[torch.amp.GradScaler]:
    """
    Build a scaler only for fp16 sink stage. For pipeline training this keeps scale
    ownership clear and avoids non-sink stages maintaining unused scaler states.
    """
    is_sink_stage = topology.local_stage_name == topology.enabled_stage_names[-1]
    if not is_sink_stage:
        return None
    if device.type != "cuda":
        return None
    if cfg.training.precision != "fp16":
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except Exception:
        # Backward compatibility path for older torch APIs.
        return torch.cuda.amp.GradScaler()


def _build_optimizer(
    cfg: RunConfig,
    topology: Topology,
    group_manager: ProcessGroupManager,
    params,
    lr: float,
) -> torch.optim.Optimizer:
    zero_stage = cfg.training.optimizer.zero_stage
    if zero_stage == 1 and topology.local_stage.dp_size > 1 and dist_ready():
        dp_group = group_manager.local_dp_group
        if dp_group is not None:
            optimizer = Zero1AdamW(
                params=params,
                lr=lr,
                weight_decay=cfg.training.optimizer.weight_decay,
                dp_group=dp_group,
                dp_global_ranks=_dp_group_global_ranks(topology),
            )
            if topology.runtime_rank == 0:
                print(
                    "[INFO] enabled ZeRO-1 distributed optimizer for stage={}, dp={}, tp={}".format(
                        topology.local_stage_name,
                        topology.local_stage.dp_size,
                        topology.local_stage.tp_size,
                    )
                )
            return optimizer
    if zero_stage == 1 and topology.runtime_rank == 0:
        print(
            "[WARN] training.optimizer.zero_stage=1 requested but not activated "
            f"for stage={topology.local_stage_name} (dp_size={topology.local_stage.dp_size}). "
            "Falling back to AdamW."
        )
    return torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=cfg.training.optimizer.weight_decay,
    )


def _dp_group_global_ranks(topology: Topology) -> list[int]:
    stage_name = topology.local_stage_name
    tp_idx = topology.local_tp_index()
    stage = topology.local_stage
    return [
        topology.rank_for(stage_name, dp_idx, tp_idx) for dp_idx in range(stage.dp_size)
    ]


def _wrap_model_with_ddp_if_needed(
    model: torch.nn.Module,
    cfg: RunConfig,
    topology: Topology,
    group_manager: ProcessGroupManager,
    device: torch.device,
) -> torch.nn.Module:
    if not dist_ready():
        return model
    if cfg.training.optimizer.zero_stage == 1:
        # ZeRO-1 distributed optimizer performs DP communication via
        # reduce-scatter/all-gather in optimizer.step(); skip DDP all-reduce hooks.
        if topology.runtime_rank == 0 and topology.local_stage.dp_size > 1:
            print(
                f"[INFO] skip DDP for stage={topology.local_stage_name} "
                "because zero_stage=1 uses distributed optimizer communication."
            )
        return model
    if topology.local_stage.dp_size <= 1:
        return model
    dp_group = group_manager.local_dp_group
    if dp_group is None:
        return model

    bucket_cap_mb = cfg.distributed.grad_sync_bucket_mb
    if bucket_cap_mb <= 0:
        bucket_cap_mb = 0.001

    kwargs = {
        "process_group": dp_group,
        "broadcast_buffers": False,
        "bucket_cap_mb": bucket_cap_mb,
        "gradient_as_bucket_view": True,
    }
    if device.type == "cuda":
        kwargs["device_ids"] = [device.index]  # type: ignore[index]
        kwargs["output_device"] = device.index

    wrapped = DDP(model, **kwargs)
    if topology.runtime_rank == 0:
        print(
            f"[INFO] enabled native DDP for stage={topology.local_stage_name}, "
            f"dp={topology.local_stage.dp_size}, bucket_cap_mb={bucket_cap_mb}"
        )
    return wrapped


def _bind_tp_group_to_model(
    model: torch.nn.Module,
    tp_group: Optional[dist.ProcessGroup],
) -> None:
    if hasattr(model, "set_tp_group"):
        model.set_tp_group(tp_group)  # type: ignore[misc]


if __name__ == "__main__":
    raise SystemExit(main())
