#!/bin/bash
#SBATCH --partition=general
#SBATCH --account=gneubig
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=iccd_wwcc
#SBATCH --output=/home/anshulk/iccd/suite_wwcc_%j.out
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
$PY -c "import accelerate" 2>/dev/null || $PIP install -q accelerate
nvidia-smi -L

$PY stage8_score.py --input suite_input_WWCC.json --output results_mistral-base.jsonl \
  --model mistralai/Mistral-Small-24B-Base-2501 --threshold -1000
# free the first model's weights before downloading the second
rm -rf "$HF_HOME"/hub/models--mistralai--Mistral-Small-24B-Base-2501 2>/dev/null || true
$PY stage8_score.py --input suite_input_WWCC.json --output results_sarvam-m.jsonl \
  --model sarvamai/sarvam-m --threshold -1000
echo "SUITE_WWCC_DONE"
