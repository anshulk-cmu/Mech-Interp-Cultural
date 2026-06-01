#!/usr/bin/env bash
# Multi-model suite scorer on a DLAMI GPU box (NN+EE combined batch).
# Usage: bash ec2_run_suite_nnee.sh <HF_TOKEN> <model1> <slug1> [<model2> <slug2> ...]
set -euo pipefail
export HF_HOME=/opt/dlami/nvme/hf   # large instance-store; root is small
source /opt/pytorch/bin/activate
pip install -q "transformers==4.48.3" "huggingface_hub<1.0" accelerate 2>&1 | tail -1 || true
export HF_TOKEN="$1"; shift
while [ "$#" -ge 2 ]; do
  MODEL="$1"; SLUG="$2"; shift 2
  echo "=== scoring $SLUG ($MODEL) ==="
  python stage8_score.py --input suite_input_NNEE.json \
    --output results_${SLUG}.jsonl --model "$MODEL" --threshold -1000
done
echo "=== suite done ==="
