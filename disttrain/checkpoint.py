from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

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
        "meta": checkpoint_metadata(config, topology),
    }
    torch.save(state, ckpt_path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    topology: Topology,
) -> int:
    ckpt_path = Path(path)
    state = torch.load(ckpt_path, map_location="cpu")
    meta = state.get("meta", {})
    _validate_checkpoint_topology(meta, topology)
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
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
