#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/_e2e_lib.sh"

STEPS="${STEPS:-3}"
FORCE_CPU="${FORCE_CPU:-1}"
TIMEOUT_SEC="${TIMEOUT_SEC:-240}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/artifacts/e2e_logs/all_topologies_${RUN_ID}}"

_e2e_check_prerequisites

echo "[INFO] Start topology regression, logs at: ${LOG_DIR}"

# 1) llm only
_e2e_run_case \
  "llm-only-2p" \
  "configs/fake_e2e_llm_only_2p.yaml" \
  "2" \
  "${STEPS}" \
  "${LOG_DIR}" \
  "${TIMEOUT_SEC}" \
  "${FORCE_CPU}"

# 2) encoder + llm
_e2e_run_case \
  "encoder-llm-2p" \
  "configs/fake_e2e_encoder_llm_2p.yaml" \
  "2" \
  "${STEPS}" \
  "${LOG_DIR}" \
  "${TIMEOUT_SEC}" \
  "${FORCE_CPU}"

# 3) llm + decoder
_e2e_run_case \
  "llm-decoder-2p" \
  "configs/fake_e2e_llm_decoder_2p.yaml" \
  "2" \
  "${STEPS}" \
  "${LOG_DIR}" \
  "${TIMEOUT_SEC}" \
  "${FORCE_CPU}"

# 4) encoder + llm + decoder
_e2e_run_case \
  "encoder-llm-decoder-3p" \
  "configs/fake_e2e_encoder_llm_decoder_3p.yaml" \
  "3" \
  "${STEPS}" \
  "${LOG_DIR}" \
  "${TIMEOUT_SEC}" \
  "${FORCE_CPU}"

echo "[INFO] All 4 topology e2e regression cases passed."
