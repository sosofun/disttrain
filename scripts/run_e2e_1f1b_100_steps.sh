#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/_e2e_lib.sh"

STEPS="${STEPS:-120}"
MIN_OPTIMIZER_STEPS="${MIN_OPTIMIZER_STEPS:-100}"
FORCE_CPU="${FORCE_CPU:-1}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1200}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/artifacts/e2e_logs/stability_1f1b_100_steps}"

_e2e_check_prerequisites

CASE_NAME="llm-only-2p-1f1b-${STEPS}steps"
_e2e_run_case \
  "${CASE_NAME}" \
  "configs/fake_e2e_llm_only_2p.yaml" \
  "2" \
  "${STEPS}" \
  "${LOG_DIR}" \
  "${TIMEOUT_SEC}" \
  "${FORCE_CPU}"

METRICS_FILE="${LOG_DIR}/${CASE_NAME}.metrics.jsonl"
python3 - "${METRICS_FILE}" "${MIN_OPTIMIZER_STEPS}" <<'PY'
import json
import math
from pathlib import Path
import sys

metrics_path = Path(sys.argv[1])
min_steps = int(sys.argv[2])
if not metrics_path.exists():
    print(f"[ERROR] metrics file not found: {metrics_path}")
    sys.exit(1)

rows = []
for line in metrics_path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    rows.append(json.loads(line))

if not rows:
    print(f"[ERROR] empty metrics file: {metrics_path}")
    sys.exit(1)

optimizer_rows = [r for r in rows if r.get("optimizer_step")]
if len(optimizer_rows) < min_steps:
    print(
        "[ERROR] insufficient optimizer-step records for stability criteria: "
        f"got={len(optimizer_rows)}, required>={min_steps}"
    )
    sys.exit(1)

for idx, r in enumerate(optimizer_rows):
    loss = float(r.get("loss", 0.0))
    if not math.isfinite(loss):
        print(f"[ERROR] non-finite loss at optimizer row idx={idx}: {loss}")
        sys.exit(1)
    grad_norm = r.get("grad_norm")
    if grad_norm is not None and not math.isfinite(float(grad_norm)):
        print(f"[ERROR] non-finite grad_norm at optimizer row idx={idx}: {grad_norm}")
        sys.exit(1)

print(
    "[INFO] 1f1b stability validation passed: "
    f"optimizer_steps={len(optimizer_rows)}, file={metrics_path}"
)
PY

echo "[INFO] 1F1B 100+ steps stability regression finished successfully."
