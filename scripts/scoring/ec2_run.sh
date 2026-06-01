#!/usr/bin/env bash
# Stage 8 runner on the DLAMI g6.xlarge. Usage: bash ec2_run.sh <HF_TOKEN>
set -euo pipefail
export HF_HOME=/opt/dlami/nvme/hf   # 229G instance store; root is only 39G and fills up
source /opt/pytorch/bin/activate
pip install -q "transformers==4.46.3" "huggingface_hub<1.0" accelerate 2>&1 | tail -1 || true
export HF_TOKEN="$1"
python stage8_score.py \
  --input  stage8_input_A01-SS-01.json \
  --output stage8_results_A01-SS-01.jsonl \
  --model  meta-llama/Llama-3.1-8B \
  --threshold 1.0
