#!/usr/bin/env bash
# Recovery runner for the EE-02 suite after Gemma-2 crashed under multi-GPU device_map='auto'
# (sliding-window KV-cache indices land on the wrong device when the model is sharded across GPUs).
# Fix: score the <=9B Gemma models on a SINGLE GPU (CUDA_VISIBLE_DEVICES=0 -> fits one L4, eager
# attn, no sharding bug); score the 24B models across all 4 GPUs. Each model is independent (|| so
# one failure can't abort the rest). llama31-base/inst already done on this box.
# Usage: bash ec2_recover_ee02.sh <HF_TOKEN>
export HF_HOME=/opt/dlami/nvme/hf
source /opt/pytorch/bin/activate
export HF_TOKEN="$1"
rm -f results_gemma2-base.jsonl results_gemma2-it.jsonl   # clear empty crashed files (resume-safe)

echo "=== gemma2-base (1 GPU) ==="
CUDA_VISIBLE_DEVICES=0 python stage8_score.py --input suite_input_EE02.json \
  --output results_gemma2-base.jsonl --model google/gemma-2-9b --threshold -1000 || echo "GEMMA-BASE FAILED"

echo "=== gemma2-it (1 GPU) ==="
CUDA_VISIBLE_DEVICES=0 python stage8_score.py --input suite_input_EE02.json \
  --output results_gemma2-it.jsonl --model google/gemma-2-9b-it --threshold -1000 || echo "GEMMA-IT FAILED"

echo "=== mistral-base (4 GPU) ==="
python stage8_score.py --input suite_input_EE02.json \
  --output results_mistral-base.jsonl --model mistralai/Mistral-Small-24B-Base-2501 --threshold -1000 || echo "MISTRAL FAILED"
rm -rf "$HF_HOME/hub"   # free the 24B shards before the next 24B (NVMe hygiene)

echo "=== sarvam-m (4 GPU) ==="
python stage8_score.py --input suite_input_EE02.json \
  --output results_sarvam-m.jsonl --model sarvamai/sarvam-m --threshold -1000 || echo "SARVAM FAILED"

echo "=== recovery done ==="
