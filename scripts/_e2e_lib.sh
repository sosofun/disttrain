#!/usr/bin/env bash
set -euo pipefail

E2E_RUNNER=()

_e2e_repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${script_dir}/.." && pwd
}

_e2e_check_prerequisites() {
  if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 not found."
    exit 1
  fi

  python3 - <<'PY'
import importlib.util
import sys

required = {
    "torch": "PyTorch",
    "yaml": "PyYAML",
}
missing = [pretty for module, pretty in required.items() if importlib.util.find_spec(module) is None]
if missing:
    print("[ERROR] Missing dependencies: " + ", ".join(missing))
    print("        Example: pip install torch pyyaml")
    sys.exit(1)
PY

  if command -v torchrun >/dev/null 2>&1; then
    E2E_RUNNER=(torchrun)
  else
    E2E_RUNNER=(python3 -m torch.distributed.run)
  fi
}

_e2e_check_log() {
  local log_file="$1"
  python3 - "$log_file" <<'PY'
from pathlib import Path
import sys

log_path = Path(sys.argv[1])
text = log_path.read_text(encoding="utf-8", errors="replace")

bad_markers = [
    "Traceback (most recent call last):",
    "RuntimeError:",
    "ChildFailedError:",
    "FAILED",
]
for marker in bad_markers:
    if marker in text:
        print(f"[ERROR] log contains failure marker: {marker}")
        sys.exit(1)

if "[step=" not in text and '"step":' not in text:
    print("[ERROR] log does not contain training step metrics (text/json).")
    sys.exit(1)

print("[INFO] log validation passed:", log_path)
PY
}

_e2e_run_case() {
  local case_name="$1"
  local config_path="$2"
  local nproc="$3"
  local steps="$4"
  local log_dir="$5"
  local timeout_sec="$6"
  local force_cpu="$7"

  mkdir -p "${log_dir}"
  local log_file="${log_dir}/${case_name}.log"
  local metrics_file="${log_dir}/${case_name}.metrics.jsonl"
  local log_format="${E2E_LOG_FORMAT:-json}"
  local cmd=(
    "${E2E_RUNNER[@]}"
    --standalone
    --nproc_per_node="${nproc}"
    train.py
    --config "${config_path}"
    --max-steps "${steps}"
    --log-format "${log_format}"
    --log-file "${metrics_file}"
  )

  echo "[INFO] >>> case=${case_name} nproc=${nproc} steps=${steps} config=${config_path}"
  echo "[INFO] >>> runner=${E2E_RUNNER[*]}"
  local rc=0
  set +e
  if [[ "${force_cpu}" == "1" ]]; then
    if command -v timeout >/dev/null 2>&1; then
      env CUDA_VISIBLE_DEVICES="" timeout "${timeout_sec}" "${cmd[@]}" 2>&1 | tee "${log_file}"
      rc=${PIPESTATUS[0]}
    else
      env CUDA_VISIBLE_DEVICES="" "${cmd[@]}" 2>&1 | tee "${log_file}"
      rc=${PIPESTATUS[0]}
    fi
  else
    if command -v timeout >/dev/null 2>&1; then
      timeout "${timeout_sec}" "${cmd[@]}" 2>&1 | tee "${log_file}"
      rc=${PIPESTATUS[0]}
    else
      "${cmd[@]}" 2>&1 | tee "${log_file}"
      rc=${PIPESTATUS[0]}
    fi
  fi
  set -e

  if [[ "${rc}" -ne 0 ]]; then
    echo "[ERROR] case=${case_name} failed with exit code ${rc}. log=${log_file}"
    exit "${rc}"
  fi
  _e2e_check_log "${log_file}"
  echo "[INFO] <<< case=${case_name} passed"
}
