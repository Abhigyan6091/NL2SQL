#!/usr/bin/env bash
# Run manually after completing all reviews and freezing the protocol.
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
"${PYTHON:-python}" -m nl2sql --config "${1:-configs/experiment.yaml}" run-experiments
