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
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/artifacts/benchmark/tp_transport_mode_compare_${RUN_ID}}"
TMP_DIR="${LOG_DIR}/tmp_configs"

_e2e_check_prerequisites
mkdir -p "${LOG_DIR}" "${TMP_DIR}"

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "[ERROR] base config not found: ${BASE_CONFIG}"
  exit 1
fi

SINGLE_CFG="${TMP_DIR}/tp_transport_single.yaml"
AUTO_CFG="${TMP_DIR}/tp_transport_auto.yaml"

python3 - "${BASE_CONFIG}" "${SINGLE_CFG}" "${AUTO_CFG}" <<'PY'
import copy
import sys
from pathlib import Path
import yaml

base = Path(sys.argv[1])
single_cfg = Path(sys.argv[2])
auto_cfg = Path(sys.argv[3])

cfg = yaml.safe_load(base.read_text(encoding="utf-8"))

cfg_single = copy.deepcopy(cfg)
cfg_single.setdefault("pipeline", {})
cfg_single["pipeline"]["transport_tp_mode"] = "single"
single_cfg.write_text(yaml.safe_dump(cfg_single, sort_keys=False), encoding="utf-8")

cfg_auto = copy.deepcopy(cfg)
cfg_auto.setdefault("pipeline", {})
cfg_auto["pipeline"]["transport_tp_mode"] = "auto"
auto_cfg.write_text(yaml.safe_dump(cfg_auto, sort_keys=False), encoding="utf-8")
PY

NPROC="$(python3 - "${BASE_CONFIG}" <<'PY'
import sys
import yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(int(cfg["distributed"]["world_size"]))
PY
)"

echo "[INFO] TP transport mode benchmark: world_size=${NPROC}, steps=${STEPS}, warmup=${WARMUP_STEPS}"

_e2e_run_case \
  "benchmark-tp-transport-single-${NPROC}p" \
  "${SINGLE_CFG}" \
  "${NPROC}" \
  "${STEPS}" \
  "${LOG_DIR}" \
  "${TIMEOUT_SEC}" \
  "${FORCE_CPU}"

_e2e_run_case \
  "benchmark-tp-transport-auto-${NPROC}p" \
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

equal_tp_boundaries = []
for i in range(len(enabled) - 1):
    lhs, rhs = enabled[i], enabled[i + 1]
    if int(cfg["stages"][lhs]["tp_size"]) == int(cfg["stages"][rhs]["tp_size"]) and int(cfg["stages"][lhs]["tp_size"]) > 1:
        equal_tp_boundaries.append(f"{lhs}->{rhs}")

files = {
    "single": log_dir / f"benchmark-tp-transport-single-{nproc}p.metrics.jsonl",
    "auto": log_dir / f"benchmark-tp-transport-auto-{nproc}p.metrics.jsonl",
}

def load_rows(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    rows = [r for r in rows if r.get("optimizer_step", False) and r.get("stage") == sink_stage]
    if warmup > 0 and len(rows) > warmup:
        rows = rows[warmup:]
    return rows

def avg(rows, key):
    vals = [float(r.get(key, 0.0)) for r in rows]
    return statistics.fmean(vals) if vals else 0.0

def sumv(rows, key):
    return float(sum(float(r.get(key, 0.0)) for r in rows))

summary = {}
for mode, path in files.items():
    if not path.exists():
        raise SystemExit(f"[ERROR] metrics file missing: {path}")
    rows = load_rows(path)
    if not rows:
        raise SystemExit(f"[ERROR] no optimizer rows for mode {mode}: {path}")
    summary[mode] = {
        "records": len(rows),
        "avg_step_time_sec": avg(rows, "step_time_sec"),
        "avg_tokens_per_sec": avg(rows, "tokens_per_sec"),
        "avg_comm_time_sec": avg(rows, "comm_time_sec"),
        "avg_comm_bytes_mb": avg(rows, "comm_bytes_mb"),
        "avg_p2p_prepost_hit_rate": avg(rows, "comm_p2p_prepost_hit_rate"),
        "avg_p2p_recv_overlap_ratio": avg(rows, "comm_p2p_recv_overlap_ratio"),
        "avg_p2p_recv_wait_sec": avg(rows, "comm_p2p_recv_wait_sec"),
        "avg_p2p_send_wait_sec": avg(rows, "comm_p2p_send_wait_sec"),
        "avg_p2p_recv_launch_sec": avg(rows, "comm_p2p_recv_launch_sec"),
        "avg_p2p_send_launch_sec": avg(rows, "comm_p2p_send_launch_sec"),
        "sum_p2p_prepost_hits": sumv(rows, "comm_p2p_prepost_hits"),
        "sum_p2p_prepost_misses": sumv(rows, "comm_p2p_prepost_misses"),
        "sum_p2p_prepost_posted": sumv(rows, "comm_p2p_prepost_posted"),
    }

single = summary["single"]
auto = summary["auto"]
report = {
    "eligible_boundaries_for_auto_direct": equal_tp_boundaries,
    "single": single,
    "auto": auto,
    "ratio_auto_vs_single_step_time": (
        auto["avg_step_time_sec"] / single["avg_step_time_sec"] if single["avg_step_time_sec"] > 0 else 0.0
    ),
    "ratio_auto_vs_single_tokens_per_sec": (
        auto["avg_tokens_per_sec"] / single["avg_tokens_per_sec"] if single["avg_tokens_per_sec"] > 0 else 0.0
    ),
    "ratio_auto_vs_single_comm_time": (
        auto["avg_comm_time_sec"] / single["avg_comm_time_sec"] if single["avg_comm_time_sec"] > 0 else 0.0
    ),
    "ratio_auto_vs_single_p2p_prepost_hit_rate": (
        auto["avg_p2p_prepost_hit_rate"] / single["avg_p2p_prepost_hit_rate"]
        if single["avg_p2p_prepost_hit_rate"] > 0
        else 0.0
    ),
    "ratio_auto_vs_single_p2p_recv_overlap_ratio": (
        auto["avg_p2p_recv_overlap_ratio"] / single["avg_p2p_recv_overlap_ratio"]
        if single["avg_p2p_recv_overlap_ratio"] > 0
        else 0.0
    ),
    "ratio_auto_vs_single_p2p_recv_wait_sec": (
        auto["avg_p2p_recv_wait_sec"] / single["avg_p2p_recv_wait_sec"]
        if single["avg_p2p_recv_wait_sec"] > 0
        else 0.0
    ),
}

json_path = log_dir / "tp_transport_mode_compare_report.json"
json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

md = []
md.append("# TP Transport Mode Benchmark Report")
md.append("")
md.append(f"- sink stage used for summary: `{sink_stage}`")
if equal_tp_boundaries:
    md.append(f"- eligible boundaries for auto direct: {', '.join(equal_tp_boundaries)}")
else:
    md.append("- eligible boundaries for auto direct: none (auto likely same as single)")
md.append("")
md.append("| Metric | Single | Auto | Auto/Single |")
md.append("|---|---:|---:|---:|")
md.append(f"| avg_step_time_sec | {single['avg_step_time_sec']:.6f} | {auto['avg_step_time_sec']:.6f} | {report['ratio_auto_vs_single_step_time']:.4f} |")
md.append(f"| avg_tokens_per_sec | {single['avg_tokens_per_sec']:.2f} | {auto['avg_tokens_per_sec']:.2f} | {report['ratio_auto_vs_single_tokens_per_sec']:.4f} |")
md.append(f"| avg_comm_time_sec | {single['avg_comm_time_sec']:.6f} | {auto['avg_comm_time_sec']:.6f} | {report['ratio_auto_vs_single_comm_time']:.4f} |")
md.append(f"| avg_comm_bytes_mb | {single['avg_comm_bytes_mb']:.4f} | {auto['avg_comm_bytes_mb']:.4f} | - |")
md.append(f"| avg_p2p_prepost_hit_rate | {single['avg_p2p_prepost_hit_rate']:.4f} | {auto['avg_p2p_prepost_hit_rate']:.4f} | {report['ratio_auto_vs_single_p2p_prepost_hit_rate']:.4f} |")
md.append(f"| avg_p2p_recv_overlap_ratio | {single['avg_p2p_recv_overlap_ratio']:.4f} | {auto['avg_p2p_recv_overlap_ratio']:.4f} | {report['ratio_auto_vs_single_p2p_recv_overlap_ratio']:.4f} |")
md.append(f"| avg_p2p_recv_wait_sec | {single['avg_p2p_recv_wait_sec']:.6f} | {auto['avg_p2p_recv_wait_sec']:.6f} | {report['ratio_auto_vs_single_p2p_recv_wait_sec']:.4f} |")
md.append(f"| avg_p2p_send_wait_sec | {single['avg_p2p_send_wait_sec']:.6f} | {auto['avg_p2p_send_wait_sec']:.6f} | - |")
md.append(f"| avg_p2p_recv_launch_sec | {single['avg_p2p_recv_launch_sec']:.6f} | {auto['avg_p2p_recv_launch_sec']:.6f} | - |")
md.append(f"| avg_p2p_send_launch_sec | {single['avg_p2p_send_launch_sec']:.6f} | {auto['avg_p2p_send_launch_sec']:.6f} | - |")
md.append("")
md.append(f"- records(single): {single['records']}")
md.append(f"- records(auto): {auto['records']}")
md.append(f"- prepost hits(single/auto): {single['sum_p2p_prepost_hits']:.0f} / {auto['sum_p2p_prepost_hits']:.0f}")
md.append(f"- prepost misses(single/auto): {single['sum_p2p_prepost_misses']:.0f} / {auto['sum_p2p_prepost_misses']:.0f}")
md.append(f"- prepost posted(single/auto): {single['sum_p2p_prepost_posted']:.0f} / {auto['sum_p2p_prepost_posted']:.0f}")

md_path = log_dir / "tp_transport_mode_compare_report.md"
md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

print(f"[INFO] report json: {json_path}")
print(f"[INFO] report md  : {md_path}")
PY

echo "[INFO] TP transport mode benchmark compare finished."
