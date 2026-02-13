from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import random
import torch

from disttrain.config import RunConfig
from disttrain.dist.topology import Topology


def checkpoint_metadata(config: RunConfig, topology: Topology) -> Dict[str, Any]:
    return {
        "enabled_stages": topology.enabled_stage_names,
        "local_stage": topology.local_stage_name,
        "stage_tp_dp": {
            stage_name: {
                "tp_size": topology.stages[stage_name].tp_size,
                "dp_size": topology.stages[stage_name].dp_size,
                "enabled": topology.stages[stage_name].enabled,
                "model_cls": topology.stages[stage_name].model_cls,
            }
            for stage_name in topology.stages
        },
        "pipeline": {
            "schedule": config.pipeline.schedule,
            "num_micro_batches": config.pipeline.num_micro_batches,
        },
        "training": {
            "hidden_size": config.training.hidden_size,
            "seq_len": config.training.seq_len,
            "vocab_size": config.training.vocab_size,
        },
    }


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    step: int,
    config: RunConfig,
    topology: Topology,
) -> None:
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": _capture_rng_state(),
        "meta": checkpoint_metadata(config, topology),
    }
    torch.save(state, ckpt_path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.amp.GradScaler],
    topology: Topology,
    restore_rng: bool = True,
) -> int:
    ckpt_path = Path(path)
    state = torch.load(ckpt_path, map_location="cpu")
    meta = state.get("meta", {})
    _validate_checkpoint_topology(meta, topology)
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    scaler_state = state.get("scaler")
    if scaler is not None and scaler_state:
        scaler.load_state_dict(scaler_state)
    if restore_rng:
        _restore_rng_state(state.get("rng"))
    return int(state.get("step", 0))


def _validate_checkpoint_topology(meta: Dict[str, Any], topology: Topology) -> None:
    ckpt_enabled = meta.get("enabled_stages", [])
    cur_enabled = topology.enabled_stage_names
    if list(ckpt_enabled) != list(cur_enabled):
        raise ValueError(
            f"checkpoint enabled_stages mismatch: ckpt={ckpt_enabled}, runtime={cur_enabled}"
        )
    ckpt_stage = meta.get("stage_tp_dp", {})
    for stage_name in cur_enabled:
        cur = topology.stages[stage_name]
        ck = ckpt_stage.get(stage_name, {})
        if int(ck.get("tp_size", -1)) != cur.tp_size or int(ck.get("dp_size", -1)) != cur.dp_size:
            raise ValueError(
                f"checkpoint stage topology mismatch at {stage_name}: "
                f"ckpt(tp,dp)=({ck.get('tp_size')},{ck.get('dp_size')}), "
                f"runtime(tp,dp)=({cur.tp_size},{cur.dp_size})"
            )


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(rng_state: Optional[Dict[str, Any]]) -> None:
    if not rng_state:
        return
    python_state = rng_state.get("python_random_state")
    if python_state is not None:
        random.setstate(python_state)
    torch_state = rng_state.get("torch_rng_state")
    if torch_state is not None:
        torch.random.set_rng_state(torch_state)
    cuda_state_all = rng_state.get("torch_cuda_rng_state_all")
    if cuda_state_all is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state_all)
