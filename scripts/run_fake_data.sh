#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-local-llm}"
STEPS="${STEPS:-3}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 not found."
  exit 1
fi

python3 - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("torch") is None:
    print("[ERROR] PyTorch is not installed. Please install torch first.")
    print("        Example: pip install torch")
    sys.exit(1)
if importlib.util.find_spec("yaml") is None:
    print("[ERROR] PyYAML is not installed. Please install pyyaml first.")
    print("        Example: pip install pyyaml")
    sys.exit(1)
PY

case "${MODE}" in
  local-llm)
    echo "[INFO] Run fake-data LLM-only local training"
    python3 train.py --config configs/fake_llm_local.yaml --max-steps "${STEPS}"
    ;;
  tri-stage-cpu)
    if ! command -v torchrun >/dev/null 2>&1; then
      echo "[ERROR] torchrun not found. Please install PyTorch distributed tools."
      exit 1
    fi
    echo "[INFO] Run fake-data tri-stage CPU distributed training (3 processes)"
    CUDA_VISIBLE_DEVICES="" torchrun --standalone --nproc_per_node=3 train.py \
      --config configs/fake_tri_stage_cpu.yaml \
      --max-steps "${STEPS}"
    ;;
  *)
    echo "Usage: $0 [local-llm|tri-stage-cpu]"
    echo "  local-llm    : single-process fake-data LLM-only training"
    echo "  tri-stage-cpu: 3-process fake-data encoder+llm+decoder training on CPU"
    exit 2
    ;;
esac
