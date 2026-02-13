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
        suggestion = _enabled_stage_mismatch_suggestion(ckpt_enabled, cur_enabled)
        raise ValueError(
            "checkpoint enabled_stages mismatch: "
            f"ckpt={ckpt_enabled}, runtime={cur_enabled}. {suggestion}"
        )
    ckpt_stage = meta.get("stage_tp_dp", {})
    for stage_name in cur_enabled:
        cur = topology.stages[stage_name]
        ck = ckpt_stage.get(stage_name, {})
        ck_tp = int(ck.get("tp_size", -1))
        ck_dp = int(ck.get("dp_size", -1))
        if ck_tp != cur.tp_size or ck_dp != cur.dp_size:
            suggestion = _stage_topology_mismatch_suggestion(
                stage_name=stage_name,
                ck_tp=ck_tp,
                ck_dp=ck_dp,
                rt_tp=cur.tp_size,
                rt_dp=cur.dp_size,
            )
            raise ValueError(
                f"checkpoint stage topology mismatch at {stage_name}: "
                f"ckpt(tp,dp)=({ck_tp},{ck_dp}), "
                f"runtime(tp,dp)=({cur.tp_size},{cur.dp_size}). {suggestion}"
            )


def _enabled_stage_mismatch_suggestion(ckpt_enabled: Any, runtime_enabled: Any) -> str:
    return (
        "建议：优先使用与 checkpoint 一致的 enabled_stages 进行恢复；"
        f"即将当前配置调整为 {list(ckpt_enabled)}。"
        "如需跨拓扑迁移，请先离线转换 checkpoint 后再 load。"
    )


def _stage_topology_mismatch_suggestion(
    stage_name: str,
    ck_tp: int,
    ck_dp: int,
    rt_tp: int,
    rt_dp: int,
) -> str:
    ck_world = ck_tp * ck_dp if ck_tp > 0 and ck_dp > 0 else -1
    rt_world = rt_tp * rt_dp if rt_tp > 0 and rt_dp > 0 else -1
    if ck_world == rt_world and ck_world > 0:
        return (
            f"建议：{stage_name} 的 stage_world_size 一致（{ck_world}），"
            "可做离线重分片映射后恢复（TP/DP 维度重排）。"
        )
    return (
        f"建议：{stage_name} 的 stage_world_size 不一致（ckpt={ck_world}, runtime={rt_world}），"
        "无法直接重映射恢复。请使用与 checkpoint 相同 tp/dp 配置恢复，"
        "或仅加载模型参数重新开始训练。"
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
