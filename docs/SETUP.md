# ICCD-6K Phase 1 — Setup & Environment Log

**Author:** Anshul Kumar, Carnegie Mellon University — anshulk@andrew.cmu.edu

Living document, updated as the build proceeds (not written at the end). Start date: 2026-05-31.
Everything below is validated against a source before being recorded. Secrets (HF token, AWS keys) are NEVER written into this repo.

## 1. Decisions locked with the user (2026-05-31)

| Topic | Decision |
|---|---|
| Conda env | Dedicated **`culture`** env (NOT base). Python **3.11**. Local env needs **no torch** — the model runs on AWS. |
| INDICA | **Taxonomy-only.** Use the locked 5-region map + 4-of-5 consensus rule from the plan (§3.2). No raw INDICA file is downloaded — no public release located, and the pipeline does not require it. |
| Stage 8 model | **Real Llama-3.1-8B base** = `meta-llama/Llama-3.1-8B` (official, gated; this token HAS access). User corrected 3 → 3.1. Same Llama-3 tokenizer as 3.0, so token-balance / F1 / F5 are unaffected. |
| Stage 8 compute | **fp16 on AWS** 24 GB EC2 (local 12 GB GPU can't hold fp16 ≈ 16 GB). Provision via AWS CLI, terminate after smoke. |
| Claude verification | Runs **via Claude Code in-session** (a subagent reviews a JSON batch, writes advisory verdicts), placed **before** human Tier-2 review. Uses the Max subscription — no API key. |
| Smoke test | One Axis-A cell: **A01-SS-01** (Regional Specificity, South, Festivals), **100** final items. Validate plumbing, then scale. |

## 2. Host environment (validated)

- **OS:** Windows 11 (`win32`). Shell: PowerShell 5.1 + Git Bash.
- **GPU (local):** NVIDIA RTX 5070 Ti Laptop, **12 GB** VRAM, driver 592.01, Blackwell (sm_120). Not used for the model (fp16 8B needs ~16 GB) → model runs on AWS. Local GPU stays free.
- **Anaconda:** `C:\Users\worka\anaconda3` (conda at `Scripts\conda.exe`). Base = Python 3.13.5, left untouched. New env `culture` (Python 3.11) created.
- **AWS CLI:** `2.34.29`, configured. Account `333650975919`, IAM user `golgi-admin`, region **us-east-1**.

## 3. HuggingFace access (validated)

- Token source: `D:\Anthropic_Fellows_Research\.env` (line starting `hf_…`). **Not copied into this repo.**
- `whoami` → **valid**, user `anshul2048`.
- Access probe (HEAD `resolve/main/config.json`):
  - `meta-llama/Llama-3.1-8B` → **200 GRANTED** ✅ (this is the reference model)
  - `meta-llama/Meta-Llama-3.1-8B` (alt name) → 401
  - `meta-llama/Meta-Llama-3-8B` (3.0) → 403 (not needed)
  - `NousResearch/Meta-Llama-3.1-8B` (public mirror) → 200 (backup, not needed)

## 4. Source datasets (validated from source)

- **SANSKRITI:** HF `13ari/Sanskriti`, file `Merged_Dataset_english_SANSKRITI.csv` (8.8 MB), license **CC0**, arXiv 2506.15355. Public, no token. → downloaded to `data/raw/sanskriti/`.
- **INDICA:** arXiv 2601.15550 confirmed. No public dataset download located → taxonomy-only (§1).
- **Wikipedia:** English MediaWiki API; per-page revision id (`oldid`) + access timestamp recorded (plan §5.3, §10.3).

## 5. AWS provisioning plan (for Stage 8, deferred until candidates are ready)

- Region **us-east-1**, default VPC `vpc-003b5ab4402aba736`. Key pairs present: `anthropic-fellows-key`, `my-asg-keypair` (need the matching `.pem` locally to SSH — TODO locate).
- Reference pattern: `D:\golgi_vcc\infrastructure\launch-instances.sh` / `teardown.sh` (golgi used **t3 CPU only** — GPU quota for this account is UNCONFIRMED).
- **Quota caveat:** `servicequotas:GetServiceQuota` is **denied** to `golgi-admin`, so the G/VT vCPU limit can't be read; a `g6.xlarge` launch may fail with `VcpuLimitExceeded` if the limit is 0. Discovered at launch; if blocked → request quota increase (out-of-band).
- Plan: **g6.xlarge** (1× L4, 24 GB, ~$0.80/hr on-demand) with a **Deep Learning OSS PyTorch GPU AMI** (CUDA + driver preinstalled). Push the ~143 Pass-A candidates, run Stage 8 fp16, pull `delta-L` back, **terminate**.

## 6. `culture` env package versions (installed 2026-05-31)

Python **3.11.15**. Full pins in `requirements-local.txt`.

| package | version | | package | version |
|---|---|---|---|---|
| huggingface_hub | 1.17.0 | | scipy | 1.17.1 |
| transformers | 5.9.0 | | statsmodels | 0.14.6 |
| tokenizers | 0.22.2 | | pandas | 3.0.3 |
| requests | 2.34.2 | | numpy | 2.4.6 |
| tqdm | 4.67.3 | | | |

No `torch` locally by design — Stage 8 runs the model on AWS. `transformers` is used CPU-side only for the Llama-3.1 tokenizer (length matching, token balance).

## 7. AWS Stage-8 run (live 2026-05-31)

- Region us-east-1c. **Instance `i-05c4da71508e102b4`** (g6.xlarge), **SG `sg-0a9590a4834a0af15`** (SSH from 98.111.206.214/32 only), key `anthropic-fellows-key`, AMI `ami-012ba162b9cd2729c` (DL PyTorch 2.7 / Ubuntu 22.04). GPU verified: **NVIDIA L4, 23 GB**. Public IP 18.234.202.156 (ephemeral).
- Workflow: scp `scripts/scoring/stage8_score.py` + the Pass-A batch → `pip install transformers huggingface_hub accelerate` → run fp16 Llama-3.1-8B scoring (`scripts/scoring/ec2_run.sh`) → scp results back → **terminate** (`scripts/scoring/teardown.ps1 -InstanceId .. -SgId ..`).
- Teardown is scripted with the exact IDs so resources are never orphaned.

## 8. Pipeline stage contracts

Per-stage I/O documented in `docs/PIPELINE.md`, updated as each stage is built.

## 9. Cross-model suite (Phase-2 preview) — see `docs/CROSSMODEL.md`

- **Model pairs (all HF-accessible with this token):** Llama-3.1-8B base/Instruct ✓✓,
  Gemma-2-9B base/it ✓✓ (Gemma access granted mid-run), Mistral-Small-24B-Base-2501 / Sarvam-M ✓✓.
- **Compute split:** 8–9B fp16 on AWS `g6e.xlarge` (L40S 48 GB); 24B fp16 on **CMU Babel** 4×A6000.
- **AWS GPU quota:** on-demand G/VT = quota `L-DB2E81BA`, **vCPUs** (g6.xlarge/g6e.xlarge=4 → fine;
  g6e.12xlarge=48 → needs increase). Granted `golgi-admin` the `ServiceQuotasFullAccess` policy (root
  console), submitted `L-DB2E81BA` 8→48 — routed to **manual support case 178026522000148** (pending).
  Babel avoids this for the 24B run.
- **Babel access:** `ssh babel`; account `gneubig`, qos `normal`, partition `general` (not `array`,
  which needs `array_qos`); weights to node-local `/data/$USER`; results to `~/iccd`. Job `8226283`.
- **Per-model scores** are appended onto `data/final/iccd_A01-SS-01.json` (`model_scores`,
  `cross_model_delta`) by `scripts/cross_model.py`.
