#!/usr/bin/env bash
# Stage-8 6-model scorer — STANDARD layout (use this as the template for future waves).
#
# Phase 1: the four <=9B models run IN PARALLEL, one per GPU (g6.12xlarge = 4xL4, ~23 GB each;
#   Llama-3.1-8B ~16 GB and Gemma-2-9B ~18.5 GB each fit a single L4) -> ~4x faster than the old
#   one-at-a-time-on-GPU-0 layout. Each worker is pinned with CUDA_VISIBLE_DEVICES so it sees one
#   physical GPU as device 0 (this also dodges the Gemma-2 multi-GPU sliding-window-cache bug).
#   `wait` barriers until all four exit, at which point their VRAM is released automatically.
# Phase 2: the two 24B models run ONE AT A TIME, each sharded across all 4 GPUs (device_map=auto).
#
# The 6 models are pre-baked in the AMI cache at $HF_HOME, so we NEVER delete that cache (the old
# `rm -rf $HF_HOME/hub` was NVMe-hygiene leftover that forced a token-gated re-download) and NO HF
# token is needed — from_pretrained loads gated models straight from local disk.
# Usage: bash ec2_run_suite_nnss04.sh [HF_TOKEN]   (token optional; unused when models are cached)
export HF_HOME=/home/ubuntu/hfcache
source /opt/pytorch/bin/activate
pip install -q "transformers==4.48.3" "huggingface_hub<1.0" accelerate 2>&1 | tail -1 || true
export HF_TOKEN="${1:-}"
IN=suite_input_NNSS04.json

# One-time HF cache-format migration upfront so the four parallel workers don't race on it.
python -c "from transformers.utils import move_cache; move_cache()" >/dev/null 2>&1 || true

echo "=== Phase 1: four <=9B models in parallel, one per GPU ($(date +%H:%M:%S)) ==="
CUDA_VISIBLE_DEVICES=0 python stage8_score.py --input "$IN" --output results_llama31-base.jsonl --model meta-llama/Llama-3.1-8B          --threshold -1000 > log_llama31-base.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python stage8_score.py --input "$IN" --output results_llama31-inst.jsonl --model meta-llama/Llama-3.1-8B-Instruct --threshold -1000 > log_llama31-inst.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python stage8_score.py --input "$IN" --output results_gemma2-base.jsonl  --model google/gemma-2-9b               --threshold -1000 > log_gemma2-base.txt  2>&1 &
CUDA_VISIBLE_DEVICES=3 python stage8_score.py --input "$IN" --output results_gemma2-it.jsonl    --model google/gemma-2-9b-it            --threshold -1000 > log_gemma2-it.txt    2>&1 &
wait
echo "=== Phase 1 done ($(date +%H:%M:%S)) ==="; wc -l results_llama31-base.jsonl results_llama31-inst.jsonl results_gemma2-base.jsonl results_gemma2-it.jsonl

echo "=== Phase 2: two 24B models sequentially, sharded over all 4 GPUs ($(date +%H:%M:%S)) ==="
python stage8_score.py --input "$IN" --output results_mistral-base.jsonl --model mistralai/Mistral-Small-24B-Base-2501 --threshold -1000 || echo "MISTRAL FAILED"
python stage8_score.py --input "$IN" --output results_sarvam-m.jsonl     --model sarvamai/sarvam-m                      --threshold -1000 || echo "SARVAM FAILED"

echo "=== NNSS04 suite done ($(date +%H:%M:%S)) ==="
wc -l results_*.jsonl