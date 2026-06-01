#!/bin/bash
#SBATCH --partition=general
#SBATCH --account=gneubig
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=iccd_nn02ss02
#SBATCH --output=/home/anshulk/iccd/suite_nn02ss02_%j.out
set -e
cd /home/anshulk/iccd
PY=/home/anshulk/miniconda3/bin/python
PIP=/home/anshulk/miniconda3/bin/pip

# node-local scratch for the 48GB weights (home has little space)
SCR=""
for d in "/data/$USER" "/scratch/$USER" "${SLURM_TMPDIR:-}" "$HOME/hfcache"; do
  [ -n "$d" ] && mkdir -p "$d/hf" 2>/dev/null && SCR="$d/hf" && break
done
export HF_HOME="$SCR"
echo "HF_HOME=$HF_HOME"; df -h "$SCR" | tail -1
export HF_TOKEN=$(cat /home/anshulk/iccd/.hf_token)
# Babel base miniconda was upgraded to huggingface_hub 1.x (httpx backend) + transformers 5.x,
# whose multi-shard downloader fails with "Cannot send a request, as the client has been closed".
# Pin to the known-good versions (matching the AWS runner) into --user site so base torch is kept
# and the change is non-destructive/reversible. PYTHONNOUSERSITE must stay unset so --user wins.
unset PYTHONNOUSERSITE
$PIP install -q --user "transformers==4.48.3" "huggingface_hub<1.0" "accelerate" 2>&1 | tail -1 || true
export HF_HUB_DOWNLOAD_TIMEOUT=60
$PY -c "import huggingface_hub,transformers; print('hub',huggingface_hub.__version__,'tf',transformers.__version__)"
nvidia-smi -L

$PY stage8_score.py --input suite_input_NN02SS02.json --output results_mistral-base.jsonl \
  --model mistralai/Mistral-Small-24B-Base-2501 --threshold -1000
# free the first model's weights before downloading the second
rm -rf "$HF_HOME"/hub/models--mistralai--Mistral-Small-24B-Base-2501 2>/dev/null || true
$PY stage8_score.py --input suite_input_NN02SS02.json --output results_sarvam-m.jsonl \
  --model sarvamai/sarvam-m --threshold -1000
echo "SUITE_NN02SS02_DONE"
