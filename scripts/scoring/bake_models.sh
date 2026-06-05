#!/usr/bin/env bash
# Pre-download all 6 Stage-8 models to a PERSISTENT root-EBS HF cache so a custom AMI captures
# them (instance-store NVMe is wiped on stop and not captured by AMIs). After this, create-image
# from the box, point launch_aws.sh at the new AMI, and set the runner's HF_HOME to /home/ubuntu/hfcache
# -> every fresh launch starts with models present (zero re-download).
# Usage: bash bake_models.sh <HF_TOKEN>
set -uo pipefail
export HF_HOME=/home/ubuntu/hfcache
export HF_TOKEN="$1"
source /opt/pytorch/bin/activate
pip install -q "huggingface_hub<1.0" 2>&1 | tail -1 || true
MODELS="meta-llama/Llama-3.1-8B meta-llama/Llama-3.1-8B-Instruct google/gemma-2-9b google/gemma-2-9b-it mistralai/Mistral-Small-24B-Base-2501 sarvamai/sarvam-m"
for r in $MODELS; do
  echo "=== DOWNLOADING $r ==="
  huggingface-cli download "$r" --exclude "original/*" "*.pth" "*.gguf" "*.bin" || echo "FAIL $r"
done
echo "ALLMODELS_DONE"
du -sh "$HF_HOME"
