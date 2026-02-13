#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/_e2e_lib.sh"

STEPS="${STEPS:-3}"
FORCE_CPU="${FORCE_CPU:-1}"
TIMEOUT_SEC="${TIMEOUT_SEC:-240}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/artifacts/e2e_logs/3stage}"

_e2e_check_prerequisites

_e2e_run_case \
  "encoder-llm-decoder-3p" \
  "configs/fake_e2e_encoder_llm_decoder_3p.yaml" \
  "3" \
  "${STEPS}" \
  "${LOG_DIR}" \
  "${TIMEOUT_SEC}" \
  "${FORCE_CPU}"

echo "[INFO] 3-stage smoke finished successfully."
