#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/_e2e_lib.sh"

BASE_CONFIG="${BASE_CONFIG:-configs/fake_tri_stage_gpu.yaml}"
STEPS="${STEPS:-8}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
TIMEOUT_SEC="${TIMEOUT_SEC:-300}"
FORCE_CPU="${FORCE_CPU:-0}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/artifacts/benchmark/sp_compare_${RUN_ID}}"
TMP_DIR="${LOG_DIR}/tmp_configs"

_e2e_check_prerequisites
mkdir -p "${LOG_DIR}" "${TMP_DIR}"

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "[ERROR] base config not found: ${BASE_CONFIG}"
  exit 1
fi

SP_ON_CFG="${TMP_DIR}/sp_on.yaml"
SP_OFF_CFG="${TMP_DIR}/sp_off.yaml"

python3 - "${BASE_CONFIG}" "${SP_ON_CFG}" "${SP_OFF_CFG}" <<'PY'
import copy
import sys
from pathlib import Path
import yaml

base = Path(sys.argv[1])
sp_on = Path(sys.argv[2])
sp_off = Path(sys.argv[3])

cfg = yaml.safe_load(base.read_text(encoding="utf-8"))

cfg_on = copy.deepcopy(cfg)
cfg_on["stages"]["llm"]["sequence_parallel"] = True
sp_on.write_text(yaml.safe_dump(cfg_on, sort_keys=False), encoding="utf-8")

cfg_off = copy.deepcopy(cfg)
cfg_off["stages"]["llm"]["sequence_parallel"] = False
sp_off.write_text(yaml.safe_dump(cfg_off, sort_keys=False), encoding="utf-8")
PY

NPROC="$(python3 - "${BASE_CONFIG}" <<'PY'
import sys
import yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(int(cfg["distributed"]["world_size"]))
PY
)"

export E2E_EXTRA_ARGS="--log-all-ranks"
_e2e_run_case \
  "benchmark-sp-on-${NPROC}p" \
  "${SP_ON_CFG}" \
  "${NPROC}" \
  "${STEPS}" \
  "${LOG_DIR}" \
  "${TIMEOUT_SEC}" \
  "${FORCE_CPU}"

_e2e_run_case \
  "benchmark-sp-off-${NPROC}p" \
  "${SP_OFF_CFG}" \
  "${NPROC}" \
  "${STEPS}" \
  "${LOG_DIR}" \
  "${TIMEOUT_SEC}" \
  "${FORCE_CPU}"
unset E2E_EXTRA_ARGS

python3 - "${LOG_DIR}" "${NPROC}" "${WARMUP_STEPS}" <<'PY'
import glob
import json
from pathlib import Path
import statistics
import sys

log_dir = Path(sys.argv[1])
nproc = int(sys.argv[2])
warmup = int(sys.argv[3])

def load_case(prefix: str):
    rows = []
    for path in sorted(glob.glob(str(log_dir / f"{prefix}.metrics.rank*.jsonl"))):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"[ERROR] no metrics rows loaded for {prefix}")
    return rows

def filter_llm_optimizer(rows):
    out = [r for r in rows if r.get("stage") == "llm" and r.get("optimizer_step")]
    if warmup > 0 and len(out) > warmup:
        out = out[warmup:]
    return out

def avg(rows, key):
    vals = [float(r.get(key, 0.0)) for r in rows]
    return statistics.fmean(vals) if vals else 0.0

on_rows = filter_llm_optimizer(load_case(f"benchmark-sp-on-{nproc}p"))
off_rows = filter_llm_optimizer(load_case(f"benchmark-sp-off-{nproc}p"))
if not on_rows or not off_rows:
    raise SystemExit("[ERROR] missing llm optimizer-step rows for SP benchmark.")

summary = {
    "sp_on": {
        "records": len(on_rows),
        "avg_gpu_mem_peak_mb": avg(on_rows, "gpu_mem_peak_mb"),
        "avg_step_time_sec": avg(on_rows, "step_time_sec"),
        "avg_tokens_per_sec": avg(on_rows, "tokens_per_sec"),
        "avg_comm_time_sec": avg(on_rows, "comm_time_sec"),
        "avg_comm_allreduce_sec": avg(on_rows, "comm_allreduce_sec"),
    },
    "sp_off": {
        "records": len(off_rows),
        "avg_gpu_mem_peak_mb": avg(off_rows, "gpu_mem_peak_mb"),
        "avg_step_time_sec": avg(off_rows, "step_time_sec"),
        "avg_tokens_per_sec": avg(off_rows, "tokens_per_sec"),
        "avg_comm_time_sec": avg(off_rows, "comm_time_sec"),
        "avg_comm_allreduce_sec": avg(off_rows, "comm_allreduce_sec"),
    },
}

on = summary["sp_on"]
off = summary["sp_off"]
summary["ratio_sp_on_vs_off_mem_peak"] = (
    on["avg_gpu_mem_peak_mb"] / off["avg_gpu_mem_peak_mb"]
    if off["avg_gpu_mem_peak_mb"] > 0 else 0.0
)
summary["ratio_sp_on_vs_off_tokens"] = (
    on["avg_tokens_per_sec"] / off["avg_tokens_per_sec"]
    if off["avg_tokens_per_sec"] > 0 else 0.0
)
summary["ratio_sp_on_vs_off_step_time"] = (
    on["avg_step_time_sec"] / off["avg_step_time_sec"]
    if off["avg_step_time_sec"] > 0 else 0.0
)

json_path = log_dir / "sp_compare_report.json"
json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

md = []
md.append("# Sequence Parallel Benchmark Report (LLM stage)")
md.append("")
md.append("| Metric | SP On | SP Off | On/Off |")
md.append("|---|---:|---:|---:|")
md.append(f"| avg_gpu_mem_peak_mb | {on['avg_gpu_mem_peak_mb']:.2f} | {off['avg_gpu_mem_peak_mb']:.2f} | {summary['ratio_sp_on_vs_off_mem_peak']:.4f} |")
md.append(f"| avg_step_time_sec | {on['avg_step_time_sec']:.6f} | {off['avg_step_time_sec']:.6f} | {summary['ratio_sp_on_vs_off_step_time']:.4f} |")
md.append(f"| avg_tokens_per_sec | {on['avg_tokens_per_sec']:.2f} | {off['avg_tokens_per_sec']:.2f} | {summary['ratio_sp_on_vs_off_tokens']:.4f} |")
md.append(f"| avg_comm_time_sec | {on['avg_comm_time_sec']:.6f} | {off['avg_comm_time_sec']:.6f} | - |")
md.append(f"| avg_comm_allreduce_sec | {on['avg_comm_allreduce_sec']:.6f} | {off['avg_comm_allreduce_sec']:.6f} | - |")
md.append("")
md.append(f"- records(sp_on): {on['records']}")
md.append(f"- records(sp_off): {off['records']}")

md_path = log_dir / "sp_compare_report.md"
md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

print(f"[INFO] report json: {json_path}")
print(f"[INFO] report md  : {md_path}")
PY

echo "[INFO] SP benchmark compare finished."
