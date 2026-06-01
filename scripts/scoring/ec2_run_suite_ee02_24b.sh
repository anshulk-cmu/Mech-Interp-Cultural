#!/usr/bin/env bash
# Multi-model suite scorer on a DLAMI GPU box -- LARGE box, the two 24B models sharded
# across 4 GPUs (g6.12xlarge = 4xL4 / 96GB, or g6e.12xlarge = 4xL40S / 192GB).
# stage8_score.py loads with device_map='auto', so a 24B fp16 model shards across the GPUs.
# The transformers==4.48.3 + huggingface_hub<1.0 pin MATTERS: hub 1.x's httpx multi-shard
# downloader fails on 24B ("Cannot send a request, as the client has been closed").
# HF cache is freed between models so the NVMe instance-store doesn't fill.
# Usage: bash ec2_run_suite_ee02_24b.sh <HF_TOKEN> <model1> <slug1> [<model2> <slug2> ...]
set -euo pipefail
export HF_HOME=/opt/dlami/nvme/hf   # large instance-store; root is small
source /opt/pytorch/bin/activate
pip install -q "transformers==4.48.3" "huggingface_hub<1.0" accelerate 2>&1 | tail -1 || true
export HF_TOKEN="$1"; shift
while [ "$#" -ge 2 ]; do
  MODEL="$1"; SLUG="$2"; shift 2
  echo "=== scoring $SLUG ($MODEL) ==="
  python stage8_score.py --input suite_input_EE02.json \
    --output results_${SLUG}.jsonl --model "$MODEL" --threshold -1000
  echo "--- freeing HF cache for $SLUG ---"
  rm -rf "${HF_HOME}/hub" || true   # wipe the 24B shards before the next model (NVMe is finite)
done
echo "=== 24B suite done ==="
