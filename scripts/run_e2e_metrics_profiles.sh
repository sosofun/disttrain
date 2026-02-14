#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/_e2e_lib.sh"

STEPS="${STEPS:-4}"
FORCE_CPU="${FORCE_CPU:-1}"
TIMEOUT_SEC="${TIMEOUT_SEC:-600}"
RUN_DIAGNOSE="${RUN_DIAGNOSE:-1}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/artifacts/e2e_logs/metrics_profiles_${RUN_ID}}"

_e2e_check_prerequisites

echo "[INFO] Start metrics profile e2e regression, logs at: ${LOG_DIR}"
echo "[INFO] STEPS=${STEPS} FORCE_CPU=${FORCE_CPU} RUN_DIAGNOSE=${RUN_DIAGNOSE}"

# 2p llm-only profiles.
_e2e_run_case \
  "metrics-llm2p-train-minimal" \
  "configs/fake_e2e_llm_only_2p_metrics_train.yaml" \
  "2" \
  "${STEPS}" \
  "${LOG_DIR}" \
  "${TIMEOUT_SEC}" \
  "${FORCE_CPU}"

_e2e_run_case \
  "metrics-llm2p-benchmark-standard" \
  "configs/fake_e2e_llm_only_2p_metrics_benchmark.yaml" \
  "2" \
  "${STEPS}" \
  "${LOG_DIR}" \
  "${TIMEOUT_SEC}" \
  "${FORCE_CPU}"

if [[ "${RUN_DIAGNOSE}" == "1" ]]; then
  _e2e_run_case \
    "metrics-llm2p-diagnose-detailed" \
    "configs/fake_e2e_llm_only_2p_metrics_diagnose.yaml" \
    "2" \
    "${STEPS}" \
    "${LOG_DIR}" \
    "${TIMEOUT_SEC}" \
    "${FORCE_CPU}"
fi

# 3p tri-stage cpu profiles.
_e2e_run_case \
  "metrics-tri3p-cpu-benchmark-standard" \
  "configs/fake_tri_stage_cpu_metrics_benchmark.yaml" \
  "3" \
  "${STEPS}" \
  "${LOG_DIR}" \
  "${TIMEOUT_SEC}" \
  "${FORCE_CPU}"

if [[ "${RUN_DIAGNOSE}" == "1" ]]; then
  _e2e_run_case \
    "metrics-tri3p-cpu-diagnose-detailed" \
    "configs/fake_tri_stage_cpu_metrics_diagnose.yaml" \
    "3" \
    "${STEPS}" \
    "${LOG_DIR}" \
    "${TIMEOUT_SEC}" \
    "${FORCE_CPU}"
fi

echo "[INFO] metrics profile e2e regression finished successfully."
