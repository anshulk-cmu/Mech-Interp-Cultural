#!/usr/bin/env bash
# Consolidated Stage-8 scorer for the A01-WW-02 (West) + A01-CC-02 (Central) Costume & Textile
# suite on ONE large box (g6.12xlarge = 4xL4). All 6 models, each independent (|| so one
# failure can't abort the rest). The <=9B models run on a SINGLE GPU (CUDA_VISIBLE_DEVICES=0):
# Gemma-2 breaks under multi-GPU device_map='auto' (sliding-window KV-cache device mismatch),
# and 8-9B fp16 fits one L4. The 24B models shard across all 4 GPUs via device_map='auto'.
# The transformers==4.48.3 + huggingface_hub<1.0 pin MATTERS for the 24B multi-shard download.
# Usage: bash ec2_run_suite_wwcc02.sh <HF_TOKEN>
export HF_HOME=/home/ubuntu/hfcache   # models pre-baked here in the iccd-stage8-6models AMI (no re-download).
                                      # On the base DLAMI instead use /opt/dlami/nvme/hf (NVMe) — but that
                                      # 200 GB root can't hold all 6 models, so prefer the models AMI.
source /opt/pytorch/bin/activate
pip install -q "transformers==4.48.3" "huggingface_hub<1.0" accelerate 2>&1 | tail -1 || true
export HF_TOKEN="$1"
IN=suite_input_WWCC02.json

echo "=== llama31-base (1 GPU) ==="
CUDA_VISIBLE_DEVICES=0 python stage8_score.py --input "$IN" \
  --output results_llama31-base.jsonl --model meta-llama/Llama-3.1-8B --threshold -1000 || echo "LLAMA-BASE FAILED"

echo "=== llama31-inst (1 GPU) ==="
CUDA_VISIBLE_DEVICES=0 python stage8_score.py --input "$IN" \
  --output results_llama31-inst.jsonl --model meta-llama/Llama-3.1-8B-Instruct --threshold -1000 || echo "LLAMA-INST FAILED"

echo "=== gemma2-base (1 GPU) ==="
CUDA_VISIBLE_DEVICES=0 python stage8_score.py --input "$IN" \
  --output results_gemma2-base.jsonl --model google/gemma-2-9b --threshold -1000 || echo "GEMMA-BASE FAILED"

echo "=== gemma2-it (1 GPU) ==="
CUDA_VISIBLE_DEVICES=0 python stage8_score.py --input "$IN" \
  --output results_gemma2-it.jsonl --model google/gemma-2-9b-it --threshold -1000 || echo "GEMMA-IT FAILED"

echo "=== mistral-base (4 GPU) ==="
python stage8_score.py --input "$IN" \
  --output results_mistral-base.jsonl --model mistralai/Mistral-Small-24B-Base-2501 --threshold -1000 || echo "MISTRAL FAILED"
rm -rf "$HF_HOME/hub"   # free the 24B shards before the next 24B (NVMe hygiene)

echo "=== sarvam-m (4 GPU) ==="
python stage8_score.py --input "$IN" \
  --output results_sarvam-m.jsonl --model sarvamai/sarvam-m --threshold -1000 || echo "SARVAM FAILED"

echo "=== WWCC02 suite done ==="
wc -l results_*.jsonl
