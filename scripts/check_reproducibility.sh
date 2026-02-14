#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/_e2e_lib.sh"

CONFIG="${CONFIG:-configs/fake_e2e_llm_only_2p.yaml}"
STEPS="${STEPS:-4}"
SEED="${SEED:-2027}"
FORCE_CPU="${FORCE_CPU:-1}"
TIMEOUT_SEC="${TIMEOUT_SEC:-300}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/artifacts/repro_check_${RUN_ID}}"
CKPT1_DIR="${LOG_DIR}/ckpt_run1"
CKPT2_DIR="${LOG_DIR}/ckpt_run2"
STRICT="${STRICT:-1}"
LOSS_ATOL="${LOSS_ATOL:-1e-12}"
GRAD_NORM_ATOL="${GRAD_NORM_ATOL:-1e-12}"

_e2e_check_prerequisites

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] config not found: ${CONFIG}"
  exit 1
fi

mkdir -p "${LOG_DIR}" "${CKPT1_DIR}" "${CKPT2_DIR}"

if [[ "${STRICT}" != "0" && "${STRICT}" != "1" ]]; then
  echo "[ERROR] STRICT must be 0 or 1, got: ${STRICT}"
  exit 1
fi

NPROC="$(python3 - "${CONFIG}" <<'PY'
import sys
import yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(int(cfg["distributed"]["world_size"]))
PY
)"

CASE1="repro-run1-${NPROC}p"
CASE2="repro-run2-${NPROC}p"

echo "[INFO] Reproducibility check start: config=${CONFIG}, nproc=${NPROC}, steps=${STEPS}, seed=${SEED}, strict=${STRICT}"

export E2E_EXTRA_ARGS="--deterministic --seed ${SEED} --checkpoint-dir ${CKPT1_DIR}"
_e2e_run_case \
  "${CASE1}" \
  "${CONFIG}" \
  "${NPROC}" \
  "${STEPS}" \
  "${LOG_DIR}" \
  "${TIMEOUT_SEC}" \
  "${FORCE_CPU}"

export E2E_EXTRA_ARGS="--deterministic --seed ${SEED} --checkpoint-dir ${CKPT2_DIR}"
_e2e_run_case \
  "${CASE2}" \
  "${CONFIG}" \
  "${NPROC}" \
  "${STEPS}" \
  "${LOG_DIR}" \
  "${TIMEOUT_SEC}" \
  "${FORCE_CPU}"
unset E2E_EXTRA_ARGS

METRICS1="${LOG_DIR}/${CASE1}.metrics.jsonl"
METRICS2="${LOG_DIR}/${CASE2}.metrics.jsonl"
REPORT_JSON="${LOG_DIR}/repro_compare_report.json"
REPORT_MD="${LOG_DIR}/repro_compare_report.md"

python3 - "${METRICS1}" "${METRICS2}" "${CKPT1_DIR}" "${CKPT2_DIR}" "${LOSS_ATOL}" "${GRAD_NORM_ATOL}" "${STRICT}" "${REPORT_JSON}" "${REPORT_MD}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

metrics1 = Path(sys.argv[1])
metrics2 = Path(sys.argv[2])
ckpt1_dir = Path(sys.argv[3])
ckpt2_dir = Path(sys.argv[4])
loss_atol = float(sys.argv[5])
grad_norm_atol = float(sys.argv[6])
strict = int(sys.argv[7]) == 1
report_json = Path(sys.argv[8])
report_md = Path(sys.argv[9])


def load_jsonl(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def sort_key(row: dict[str, Any]):
    return (
        int(row.get("step", -1)),
        int(row.get("rank", -1)),
        str(row.get("stage", "")),
        int(row.get("local_tp_idx", -1)),
        int(row.get("local_dp_idx", -1)),
        bool(row.get("optimizer_step", False)),
    )


def assert_close(a: Any, b: Any, atol: float, name: str, idx: int) -> None:
    if a is None and b is None:
        return
    if a is None or b is None:
        raise SystemExit(
            f"[ERROR] metric mismatch at row={idx}, key={name}: {a} != {b}"
        )
    fa = float(a)
    fb = float(b)
    if not math.isfinite(fa) or not math.isfinite(fb):
        raise SystemExit(
            f"[ERROR] non-finite metric at row={idx}, key={name}: {fa}, {fb}"
        )
    if abs(fa - fb) > atol:
        raise SystemExit(
            f"[ERROR] metric mismatch at row={idx}, key={name}: "
            f"{fa} vs {fb}, atol={atol}"
        )


def compare_metrics(rows1: list[dict[str, Any]], rows2: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows1) != len(rows2):
        raise SystemExit(
            f"[ERROR] metrics length mismatch: run1={len(rows1)}, run2={len(rows2)}"
        )
    rows1 = sorted(rows1, key=sort_key)
    rows2 = sorted(rows2, key=sort_key)

    exact_keys = [
        "step",
        "rank",
        "stage",
        "local_tp_idx",
        "local_dp_idx",
        "sequence_parallel",
        "sync_impl",
        "optimizer_step",
    ]
    for i, (a, b) in enumerate(zip(rows1, rows2)):
        for key in exact_keys:
            if a.get(key) != b.get(key):
                raise SystemExit(
                    f"[ERROR] metric mismatch at row={i}, key={key}: {a.get(key)} != {b.get(key)}"
                )
        # Strict mode validates floating-point metric reproducibility.
        # Non-strict mode only requires structural metrics + checkpoint digests.
        if strict:
            assert_close(a.get("loss"), b.get("loss"), loss_atol, "loss", i)
            assert_close(a.get("grad_norm"), b.get("grad_norm"), grad_norm_atol, "grad_norm", i)
            assert_close(a.get("lr"), b.get("lr"), 0.0, "lr", i)
            assert_close(a.get("scaler_scale"), b.get("scaler_scale"), 0.0, "scaler_scale", i)

    optimizer_steps = sum(1 for r in rows1 if r.get("optimizer_step"))
    return {
        "rows": len(rows1),
        "optimizer_rows": optimizer_steps,
        "strict": strict,
        "float_metric_check": "enabled" if strict else "skipped",
        "loss_atol": loss_atol,
        "grad_norm_atol": grad_norm_atol,
    }


def _hash_tensor(t: torch.Tensor, hasher: "hashlib._Hash") -> None:
    cpu = t.detach().cpu().contiguous()
    hasher.update(str(cpu.dtype).encode("utf-8"))
    hasher.update(str(tuple(cpu.shape)).encode("utf-8"))
    hasher.update(cpu.view(torch.uint8).numpy().tobytes())


def _hash_obj(obj: Any, hasher: "hashlib._Hash") -> None:
    if isinstance(obj, torch.Tensor):
        hasher.update(b"T")
        _hash_tensor(obj, hasher)
        return
    if isinstance(obj, dict):
        hasher.update(b"D")
        for k in sorted(obj.keys(), key=lambda x: str(x)):
            hasher.update(str(k).encode("utf-8"))
            _hash_obj(obj[k], hasher)
        return
    if isinstance(obj, (list, tuple)):
        hasher.update(b"L")
        for item in obj:
            _hash_obj(item, hasher)
        return
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        hasher.update(repr(obj).encode("utf-8"))
        return
    # Fallback for rare objects (e.g. RNG tuples).
    hasher.update(repr(obj).encode("utf-8"))


def ckpt_digest(path: Path) -> str:
    state = torch.load(path, map_location="cpu")
    hasher = hashlib.sha256()
    _hash_obj(state.get("step"), hasher)
    _hash_obj(state.get("model"), hasher)
    _hash_obj(state.get("optimizer"), hasher)
    _hash_obj(state.get("scaler"), hasher)
    _hash_obj(state.get("rng"), hasher)
    _hash_obj(state.get("meta"), hasher)
    return hasher.hexdigest()


def compare_checkpoints(dir1: Path, dir2: Path) -> dict[str, Any]:
    files1 = sorted(dir1.glob("*.pt"))
    files2 = sorted(dir2.glob("*.pt"))
    names1 = [p.name for p in files1]
    names2 = [p.name for p in files2]
    if names1 != names2:
        raise SystemExit(
            f"[ERROR] checkpoint file set mismatch: run1={names1}, run2={names2}"
        )
    if not files1:
        raise SystemExit("[ERROR] no checkpoint files found for reproducibility comparison")

    digests = {}
    for p1, p2 in zip(files1, files2):
        d1 = ckpt_digest(p1)
        d2 = ckpt_digest(p2)
        if d1 != d2:
            raise SystemExit(
                f"[ERROR] checkpoint digest mismatch for {p1.name}: run1={d1}, run2={d2}"
            )
        digests[p1.name] = d1
    return {"count": len(digests), "digests": digests}


if not metrics1.exists() or not metrics2.exists():
    raise SystemExit(f"[ERROR] metrics file missing: {metrics1} / {metrics2}")

rows1 = load_jsonl(metrics1)
rows2 = load_jsonl(metrics2)
if not rows1 or not rows2:
    raise SystemExit("[ERROR] metrics rows are empty, cannot compare reproducibility")

metrics_summary = compare_metrics(rows1, rows2)
ckpt_summary = compare_checkpoints(ckpt1_dir, ckpt2_dir)

report = {
    "metrics": metrics_summary,
    "checkpoints": ckpt_summary,
    "status": "passed",
}
report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

md = [
    "# Reproducibility Check Report",
    "",
    "## Result",
    "",
    "- status: passed",
    f"- strict mode: {metrics_summary['strict']}",
    f"- float metric check: {metrics_summary['float_metric_check']}",
    f"- metrics rows: {metrics_summary['rows']}",
    f"- optimizer rows: {metrics_summary['optimizer_rows']}",
    f"- checkpoint files compared: {ckpt_summary['count']}",
    "",
    "## Metrics Tolerance",
    "",
    f"- loss_atol: {metrics_summary['loss_atol']}",
    f"- grad_norm_atol: {metrics_summary['grad_norm_atol']}",
]
report_md.write_text("\n".join(md) + "\n", encoding="utf-8")

print(f"[INFO] reproducibility report json: {report_json}")
print(f"[INFO] reproducibility report md  : {report_md}")
print("[INFO] reproducibility check passed.")
PY

echo "[INFO] Reproducibility check finished successfully."
