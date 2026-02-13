#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/_e2e_lib.sh"

STEPS="${STEPS:-3}"
FORCE_CPU="${FORCE_CPU:-1}"
TIMEOUT_SEC="${TIMEOUT_SEC:-180}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/artifacts/e2e_logs/ddp_dp_sync}"

_e2e_check_prerequisites

CASE_NAME="llm-only-2p-ddp"
_e2e_run_case \
  "${CASE_NAME}" \
  "configs/fake_e2e_llm_only_2p.yaml" \
  "2" \
  "${STEPS}" \
  "${LOG_DIR}" \
  "${TIMEOUT_SEC}" \
  "${FORCE_CPU}"

METRICS_FILE="${LOG_DIR}/${CASE_NAME}.metrics.jsonl"
python3 - "${METRICS_FILE}" <<'PY'
import json
from pathlib import Path
import sys

metrics_path = Path(sys.argv[1])
if not metrics_path.exists():
    print(f"[ERROR] metrics file not found: {metrics_path}")
    sys.exit(1)

records = []
for line in metrics_path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    records.append(json.loads(line))

if not records:
    print(f"[ERROR] metrics file is empty: {metrics_path}")
    sys.exit(1)

optimizer_steps = [r for r in records if r.get("optimizer_step")]
if not optimizer_steps:
    print("[ERROR] no optimizer step records found in metrics.")
    sys.exit(1)

if not any(str(r.get("sync_impl", "")).startswith("ddp") for r in optimizer_steps):
    print("[ERROR] DDP sync path not observed in optimizer-step records.")
    print("        expected sync_impl to start with 'ddp'.")
    sys.exit(1)

print(f"[INFO] DDP sync validation passed: {metrics_path}")
PY

echo "[INFO] DDP DP-sync e2e regression finished successfully."
