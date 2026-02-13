from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import random
import os
import json

import torch
import torch.distributed as dist

from disttrain.checkpoint import load_checkpoint, save_checkpoint
from disttrain.config import ConfigError, RunConfig, load_config
from disttrain.dist.groups import ProcessGroupManager
from disttrain.dist.topology import Topology, TopologyError
from disttrain.models.registry import build_stage_model
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
    return parser.parse_args()


def dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_seed(seed: int, rank: int) -> None:
    final_seed = seed + rank
    random.seed(final_seed)
    torch.manual_seed(final_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(final_seed)


def init_distributed(config: RunConfig) -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and not dist_ready():
        backend = config.distributed.backend
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


def main() -> int:
    args = parse_args()

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"[ERROR] config error: {exc}")
        return 2

    world_size, rank, _, device = init_distributed(cfg)
    setup_seed(args.seed, rank)

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
    model = build_stage_model(stage_cfg, cfg.training).to(device)
    stage_lr = cfg.training.optimizer.stage_lrs.get(
        topology.local_stage_name,
        cfg.training.optimizer.lr,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=stage_lr,
        weight_decay=cfg.training.optimizer.weight_decay,
    )

    group_manager = ProcessGroupManager(topology)
    group_manager.create()
    group_manager.sync_parameters(model)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, topology)
        if is_log_rank(topology):
            print(f"[INFO][rank={rank}] resumed from {args.resume}, step={start_step}")

    engine = TrainingEngine(
        config=cfg,
        topology=topology,
        group_manager=group_manager,
        model=model,
        optimizer=optimizer,
        device=device,
    )

    max_steps = args.max_steps if args.max_steps is not None else cfg.training.max_steps
    if max_steps <= start_step:
        max_steps = start_step + 1

    metrics = engine.run(max_steps=max_steps - start_step)
    if is_log_rank(topology):
        for m in metrics:
            if args.log_format == "json":
                payload = {
                    "step": start_step + m.step,
                    "rank": rank,
                    "stage": topology.local_stage_name,
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
                    "gpu_mem_peak_mb": m.gpu_mem_peak_mb,
                    "grad_norm": m.grad_norm,
                    "lr": m.lr,
                    "optimizer_step": m.optimizer_stepped,
                }
                print(json.dumps(payload, ensure_ascii=False))
            else:
                print(
                    "[step={:04d}] loss={:.6f} step_time={:.3f}s fwd={:.3f}s bwd={:.3f}s "
                    "tokens/s={:.1f} samples/s={:.1f} comm={:.3f}s bw={:.2f}MB/s "
                    "grad_norm={} lr={:.6g} bubble={:.4f} optimizer_step={}".format(
                        start_step + m.step,
                        m.loss,
                        m.step_time_sec,
                        m.forward_time_sec,
                        m.backward_time_sec,
                        m.tokens_per_sec,
                        m.samples_per_sec,
                        m.comm_time_sec,
                        m.comm_bandwidth_mb_s,
                        "n/a" if m.grad_norm is None else f"{m.grad_norm:.4f}",
                        m.lr,
                        m.bubble_ratio,
                        m.optimizer_stepped,
                    )
                )

    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / (
            f"stage_{topology.local_stage_name}_rank_{topology.runtime_rank}.pt"
        )
        save_checkpoint(
            path=str(ckpt_path),
            model=model,
            optimizer=optimizer,
            step=start_step + len(metrics),
            config=cfg,
            topology=topology,
        )
        if is_log_rank(topology):
            print(f"[INFO][rank={rank}] checkpoint saved: {ckpt_path}")

    group_manager.clear()
    if dist_ready():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
