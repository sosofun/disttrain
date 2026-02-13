#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/_e2e_lib.sh"

STEPS="${STEPS:-3}"
CASE="${1:-all}"   # all | encoder-llm | llm-decoder
FORCE_CPU="${FORCE_CPU:-1}"
TIMEOUT_SEC="${TIMEOUT_SEC:-180}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/artifacts/e2e_logs/2stage}"

_e2e_check_prerequisites

case "${CASE}" in
  encoder-llm)
    _e2e_run_case \
      "encoder-llm-2p" \
      "configs/fake_e2e_encoder_llm_2p.yaml" \
      "2" \
      "${STEPS}" \
      "${LOG_DIR}" \
      "${TIMEOUT_SEC}" \
      "${FORCE_CPU}"
    ;;
  llm-decoder)
    _e2e_run_case \
      "llm-decoder-2p" \
      "configs/fake_e2e_llm_decoder_2p.yaml" \
      "2" \
      "${STEPS}" \
      "${LOG_DIR}" \
      "${TIMEOUT_SEC}" \
      "${FORCE_CPU}"
    ;;
  all)
    _e2e_run_case \
      "encoder-llm-2p" \
      "configs/fake_e2e_encoder_llm_2p.yaml" \
      "2" \
      "${STEPS}" \
      "${LOG_DIR}" \
      "${TIMEOUT_SEC}" \
      "${FORCE_CPU}"
    _e2e_run_case \
      "llm-decoder-2p" \
      "configs/fake_e2e_llm_decoder_2p.yaml" \
      "2" \
      "${STEPS}" \
      "${LOG_DIR}" \
      "${TIMEOUT_SEC}" \
      "${FORCE_CPU}"
    ;;
  *)
    echo "Usage: $0 [all|encoder-llm|llm-decoder]"
    exit 2
    ;;
esac

echo "[INFO] 2-stage smoke finished successfully."
