#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/_e2e_lib.sh"

BASE_CONFIG="${BASE_CONFIG:-configs/fake_tri_stage_gpu.yaml}"
STEPS="${STEPS:-8}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
FORCE_CPU="${FORCE_CPU:-0}"
TIMEOUT_SEC="${TIMEOUT_SEC:-300}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/artifacts/benchmark/transport_dtype_compare_${RUN_ID}}"
TMP_DIR="${LOG_DIR}/tmp_configs"

_e2e_check_prerequisites
mkdir -p "${LOG_DIR}" "${TMP_DIR}"

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "[ERROR] base config not found: ${BASE_CONFIG}"
  exit 1
fi

FP32_CFG="${TMP_DIR}/transport_fp32.yaml"
AUTO_CFG="${TMP_DIR}/transport_auto.yaml"

python3 - "${BASE_CONFIG}" "${FP32_CFG}" "${AUTO_CFG}" <<'PY'
import copy
import sys
from pathlib import Path
import yaml

base = Path(sys.argv[1])
fp32_cfg = Path(sys.argv[2])
auto_cfg = Path(sys.argv[3])

cfg = yaml.safe_load(base.read_text(encoding="utf-8"))

cfg_fp32 = copy.deepcopy(cfg)
cfg_fp32.setdefault("pipeline", {})
cfg_fp32["pipeline"]["transport_dtype"] = "fp32"
fp32_cfg.write_text(yaml.safe_dump(cfg_fp32, sort_keys=False), encoding="utf-8")

cfg_auto = copy.deepcopy(cfg)
cfg_auto.setdefault("pipeline", {})
cfg_auto["pipeline"]["transport_dtype"] = "auto"
auto_cfg.write_text(yaml.safe_dump(cfg_auto, sort_keys=False), encoding="utf-8")
PY

NPROC="$(python3 - "${BASE_CONFIG}" <<'PY'
import sys
import yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(int(cfg["distributed"]["world_size"]))
PY
)"

echo "[INFO] Transport dtype benchmark: world_size=${NPROC}, steps=${STEPS}, warmup=${WARMUP_STEPS}"

_e2e_run_case \
  "benchmark-transport-fp32-${NPROC}p" \
  "${FP32_CFG}" \
  "${NPROC}" \
  "${STEPS}" \
  "${LOG_DIR}" \
  "${TIMEOUT_SEC}" \
  "${FORCE_CPU}"

_e2e_run_case \
  "benchmark-transport-auto-${NPROC}p" \
  "${AUTO_CFG}" \
  "${NPROC}" \
  "${STEPS}" \
  "${LOG_DIR}" \
  "${TIMEOUT_SEC}" \
  "${FORCE_CPU}"

python3 - "${LOG_DIR}" "${BASE_CONFIG}" "${NPROC}" "${WARMUP_STEPS}" <<'PY'
import json
from pathlib import Path
import statistics
import sys
import yaml

log_dir = Path(sys.argv[1])
base_cfg = Path(sys.argv[2])
nproc = int(sys.argv[3])
warmup = int(sys.argv[4])

cfg = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
enabled = [s for s in ("encoder", "llm", "decoder") if cfg.get("stages", {}).get(s, {}).get("enabled", False)]
sink_stage = enabled[-1] if enabled else "llm"

files = {
    "fp32": log_dir / f"benchmark-transport-fp32-{nproc}p.metrics.jsonl",
    "auto": log_dir / f"benchmark-transport-auto-{nproc}p.metrics.jsonl",
}

def load_rows(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    rows = [r for r in rows if r.get("optimizer_step", False) and r.get("stage") == sink_stage]
    if not rows:
        # Fallback to any optimizer rows if sink-only rows were not logged.
        all_rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            all_rows.append(json.loads(line))
        rows = [r for r in all_rows if r.get("optimizer_step", False)]
    if warmup > 0 and len(rows) > warmup:
        rows = rows[warmup:]
    return rows

def avg(rows, key):
    vals = [float(r.get(key, 0.0)) for r in rows]
    return statistics.fmean(vals) if vals else 0.0

summary = {}
for mode, path in files.items():
    if not path.exists():
        raise SystemExit(f"[ERROR] metrics file missing: {path}")
    rows = load_rows(path)
    if not rows:
        raise SystemExit(f"[ERROR] no optimizer rows for transport mode {mode}: {path}")
    summary[mode] = {
        "records": len(rows),
        "avg_step_time_sec": avg(rows, "step_time_sec"),
        "avg_tokens_per_sec": avg(rows, "tokens_per_sec"),
        "avg_comm_time_sec": avg(rows, "comm_time_sec"),
        "avg_comm_bytes_mb": avg(rows, "comm_bytes_mb"),
        "avg_comm_bandwidth_mb_s": avg(rows, "comm_bandwidth_mb_s"),
    }

fp32 = summary["fp32"]
auto = summary["auto"]
report = {
    "fp32": fp32,
    "auto": auto,
    "ratio_auto_vs_fp32_step_time": (
        auto["avg_step_time_sec"] / fp32["avg_step_time_sec"] if fp32["avg_step_time_sec"] > 0 else 0.0
    ),
    "ratio_auto_vs_fp32_tokens_per_sec": (
        auto["avg_tokens_per_sec"] / fp32["avg_tokens_per_sec"] if fp32["avg_tokens_per_sec"] > 0 else 0.0
    ),
    "ratio_auto_vs_fp32_comm_time": (
        auto["avg_comm_time_sec"] / fp32["avg_comm_time_sec"] if fp32["avg_comm_time_sec"] > 0 else 0.0
    ),
}

json_path = log_dir / "transport_dtype_compare_report.json"
json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

md = []
md.append("# Transport DType Benchmark Report")
md.append("")
md.append(f"- sink stage used for summary: `{sink_stage}`")
md.append("")
md.append("| Metric | FP32 | Auto | Auto/FP32 |")
md.append("|---|---:|---:|---:|")
md.append(f"| avg_step_time_sec | {fp32['avg_step_time_sec']:.6f} | {auto['avg_step_time_sec']:.6f} | {report['ratio_auto_vs_fp32_step_time']:.4f} |")
md.append(f"| avg_tokens_per_sec | {fp32['avg_tokens_per_sec']:.2f} | {auto['avg_tokens_per_sec']:.2f} | {report['ratio_auto_vs_fp32_tokens_per_sec']:.4f} |")
md.append(f"| avg_comm_time_sec | {fp32['avg_comm_time_sec']:.6f} | {auto['avg_comm_time_sec']:.6f} | {report['ratio_auto_vs_fp32_comm_time']:.4f} |")
md.append(f"| avg_comm_bytes_mb | {fp32['avg_comm_bytes_mb']:.4f} | {auto['avg_comm_bytes_mb']:.4f} | - |")
md.append(f"| avg_comm_bandwidth_mb_s | {fp32['avg_comm_bandwidth_mb_s']:.2f} | {auto['avg_comm_bandwidth_mb_s']:.2f} | - |")
md.append("")
md.append(f"- records(fp32): {fp32['records']}")
md.append(f"- records(auto): {auto['records']}")

md_path = log_dir / "transport_dtype_compare_report.md"
md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

print(f"[INFO] report json: {json_path}")
print(f"[INFO] report md  : {md_path}")
PY

echo "[INFO] Transport dtype benchmark compare finished."
