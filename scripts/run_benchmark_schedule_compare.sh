#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/_e2e_lib.sh"

BASE_CONFIG="${BASE_CONFIG:-configs/fake_benchmark_schedule_3p.yaml}"
STEPS="${STEPS:-8}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
FORCE_CPU="${FORCE_CPU:-1}"
TIMEOUT_SEC="${TIMEOUT_SEC:-300}"
REQUIRE_1F1B_NOT_WORSE="${REQUIRE_1F1B_NOT_WORSE:-1}"
TOKENS_RATIO_MIN="${TOKENS_RATIO_MIN:-0.98}"
STEP_TIME_RATIO_MAX="${STEP_TIME_RATIO_MAX:-1.02}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/artifacts/benchmark/schedule_compare_${RUN_ID}}"
TMP_DIR="${LOG_DIR}/tmp_configs"

_e2e_check_prerequisites
mkdir -p "${LOG_DIR}" "${TMP_DIR}"

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "[ERROR] base config not found: ${BASE_CONFIG}"
  exit 1
fi

GPipe_CFG="${TMP_DIR}/gpipe.yaml"
F1B_CFG="${TMP_DIR}/1f1b.yaml"

python3 - "${BASE_CONFIG}" "${GPipe_CFG}" "${F1B_CFG}" <<'PY'
import sys
from pathlib import Path
import yaml

base = Path(sys.argv[1])
gpipe_path = Path(sys.argv[2])
f1b_path = Path(sys.argv[3])
cfg = yaml.safe_load(base.read_text(encoding="utf-8"))

cfg_g = dict(cfg)
cfg_g["pipeline"] = dict(cfg["pipeline"])
cfg_g["pipeline"]["schedule"] = "gpipe"
gpipe_path.write_text(yaml.safe_dump(cfg_g, sort_keys=False), encoding="utf-8")

cfg_f = dict(cfg)
cfg_f["pipeline"] = dict(cfg["pipeline"])
cfg_f["pipeline"]["schedule"] = "1f1b"
f1b_path.write_text(yaml.safe_dump(cfg_f, sort_keys=False), encoding="utf-8")
PY

NPROC="$(python3 - "${BASE_CONFIG}" <<'PY'
import sys
import yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(int(cfg["distributed"]["world_size"]))
PY
)"

echo "[INFO] Benchmark schedule compare: world_size=${NPROC}, steps=${STEPS}, warmup=${WARMUP_STEPS}"

_e2e_run_case \
  "benchmark-gpipe-${NPROC}p" \
  "${GPipe_CFG}" \
  "${NPROC}" \
  "${STEPS}" \
  "${LOG_DIR}" \
  "${TIMEOUT_SEC}" \
  "${FORCE_CPU}"

_e2e_run_case \
  "benchmark-1f1b-${NPROC}p" \
  "${F1B_CFG}" \
  "${NPROC}" \
  "${STEPS}" \
  "${LOG_DIR}" \
  "${TIMEOUT_SEC}" \
  "${FORCE_CPU}"

python3 - "${LOG_DIR}" "${NPROC}" "${WARMUP_STEPS}" "${REQUIRE_1F1B_NOT_WORSE}" "${TOKENS_RATIO_MIN}" "${STEP_TIME_RATIO_MAX}" <<'PY'
import json
from pathlib import Path
import statistics
import sys

log_dir = Path(sys.argv[1])
nproc = int(sys.argv[2])
warmup = int(sys.argv[3])
require_gate = int(sys.argv[4]) == 1
tokens_ratio_min = float(sys.argv[5])
step_time_ratio_max = float(sys.argv[6])

files = {
    "gpipe": log_dir / f"benchmark-gpipe-{nproc}p.metrics.jsonl",
    "1f1b": log_dir / f"benchmark-1f1b-{nproc}p.metrics.jsonl",
}

def load_records(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    rows = [r for r in rows if r.get("optimizer_step", False)]
    if warmup > 0 and len(rows) > warmup:
        rows = rows[warmup:]
    return rows

def avg(rows, key):
    vals = [float(r.get(key, 0.0)) for r in rows]
    return statistics.fmean(vals) if vals else 0.0

summary = {}
for sched, path in files.items():
    if not path.exists():
        raise SystemExit(f"[ERROR] metrics file missing: {path}")
    rows = load_records(path)
    if not rows:
        raise SystemExit(f"[ERROR] no optimizer-step rows in metrics: {path}")
    summary[sched] = {
        "records": len(rows),
        "avg_step_time_sec": avg(rows, "step_time_sec"),
        "avg_tokens_per_sec": avg(rows, "tokens_per_sec"),
        "avg_samples_per_sec": avg(rows, "samples_per_sec"),
        "avg_comm_time_sec": avg(rows, "comm_time_sec"),
        "avg_comm_allreduce_sec": avg(rows, "comm_allreduce_sec"),
        "avg_dataloader_wait_sec": avg(rows, "dataloader_wait_sec"),
        "avg_host_to_device_sec": avg(rows, "host_to_device_sec"),
    }

g = summary["gpipe"]
f = summary["1f1b"]
report = {
    "gpipe": g,
    "1f1b": f,
    "ratio_1f1b_vs_gpipe_tokens_per_sec": (
        f["avg_tokens_per_sec"] / g["avg_tokens_per_sec"] if g["avg_tokens_per_sec"] > 0 else 0.0
    ),
    "ratio_1f1b_vs_gpipe_samples_per_sec": (
        f["avg_samples_per_sec"] / g["avg_samples_per_sec"] if g["avg_samples_per_sec"] > 0 else 0.0
    ),
    "ratio_1f1b_vs_gpipe_step_time": (
        f["avg_step_time_sec"] / g["avg_step_time_sec"] if g["avg_step_time_sec"] > 0 else 0.0
    ),
}

json_path = log_dir / "schedule_compare_report.json"
json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

md = []
md.append("# Schedule Benchmark Report")
md.append("")
md.append("| Metric | GPipe | 1F1B | 1F1B/GPipe |")
md.append("|---|---:|---:|---:|")
md.append(f"| avg_step_time_sec | {g['avg_step_time_sec']:.6f} | {f['avg_step_time_sec']:.6f} | {report['ratio_1f1b_vs_gpipe_step_time']:.4f} |")
md.append(f"| avg_tokens_per_sec | {g['avg_tokens_per_sec']:.2f} | {f['avg_tokens_per_sec']:.2f} | {report['ratio_1f1b_vs_gpipe_tokens_per_sec']:.4f} |")
md.append(f"| avg_samples_per_sec | {g['avg_samples_per_sec']:.2f} | {f['avg_samples_per_sec']:.2f} | {report['ratio_1f1b_vs_gpipe_samples_per_sec']:.4f} |")
md.append(f"| avg_comm_time_sec | {g['avg_comm_time_sec']:.6f} | {f['avg_comm_time_sec']:.6f} | - |")
md.append(f"| avg_comm_allreduce_sec | {g['avg_comm_allreduce_sec']:.6f} | {f['avg_comm_allreduce_sec']:.6f} | - |")
md.append(f"| avg_dataloader_wait_sec | {g['avg_dataloader_wait_sec']:.6f} | {f['avg_dataloader_wait_sec']:.6f} | - |")
md.append(f"| avg_host_to_device_sec | {g['avg_host_to_device_sec']:.6f} | {f['avg_host_to_device_sec']:.6f} | - |")
md.append("")
md.append(f"- records(gpipe): {g['records']}")
md.append(f"- records(1f1b): {f['records']}")

md_path = log_dir / "schedule_compare_report.md"
md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

print(f"[INFO] report json: {json_path}")
print(f"[INFO] report md  : {md_path}")

if require_gate:
    if report["ratio_1f1b_vs_gpipe_tokens_per_sec"] < tokens_ratio_min:
        print(
            "[ERROR] benchmark gate failed: 1f1b tokens/s ratio is below threshold: "
            f"{report['ratio_1f1b_vs_gpipe_tokens_per_sec']:.4f} < {tokens_ratio_min:.4f}"
        )
        sys.exit(2)
    if report["ratio_1f1b_vs_gpipe_step_time"] > step_time_ratio_max:
        print(
            "[ERROR] benchmark gate failed: 1f1b step-time ratio is above threshold: "
            f"{report['ratio_1f1b_vs_gpipe_step_time']:.4f} > {step_time_ratio_max:.4f}"
        )
        sys.exit(2)
    print(
        "[INFO] benchmark gate passed: "
        f"tokens_ratio={report['ratio_1f1b_vs_gpipe_tokens_per_sec']:.4f}, "
        f"step_time_ratio={report['ratio_1f1b_vs_gpipe_step_time']:.4f}"
    )
PY

echo "[INFO] Schedule benchmark compare finished."
