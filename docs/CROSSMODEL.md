# Cross-Model Evaluation — base↔aligned delta-L signal (A01-SS-01)

**Author:** Anshul Kumar, Carnegie Mellon University — anshulk@andrew.cmu.edu

The project's question is **selectivity located mechanistically**: where in the network alignment
reshapes cultural knowledge and whether that change is recoverable — with **rewrite-vs-gate** as the
mechanism that explains the selectivity. This page is a Phase-2 **preview** on the finished clean 100
(cell A01-SS-01). For each base↔aligned model pair we score the per-item **delta-L** =
logP(target|clean) − logP(target|corrupted) (Zhang & Nanda logit-difference, nats), then the
cross-model signal:

```
delta(item) = delta-L_base − delta-L_aligned
```

`delta ≈ 0` → alignment left the festival→state binding intact (representation preserved).
`delta ≫ 0` → base binds more strongly than the aligned model (candidate gating/rewrite).
`delta < 0` → the aligned model is *more* confident (alignment sharpened the benign fact).

All six per-model scores are appended onto each item in `data/final/iccd_A01-SS-01.json`
(`model_scores` = {delta_L, logp_clean, logp_corrupt} per model; `cross_model_delta` = per pair).

> **Scope of this preview (read before the numbers).** A01-SS-01 is a **single cell**: it has **zero
> axis variation and zero region variation**. It therefore **cannot evidence the selectivity
> interaction (axis×region / axis×model-pair)**, which is the project's headline test — that requires
> the full cell grid, not one cell. Nothing below demonstrates selectivity or settles a mechanism
> verdict; this is a directional preview of the cross-recipe question, not a result. This is a
> diagnostic, not a benchmark.

## The three pairs (all accessible)

| Pair | Base | Aligned | Params | Alignment |
|---|---|---|---|---|
| Llama-3.1-8B | meta-llama/Llama-3.1-8B | meta-llama/Llama-3.1-8B-Instruct | 8B | RLHF (instruct) |
| Gemma-2-9B | google/gemma-2-9b | google/gemma-2-9b-it | 9B | RLHF (it) |
| Mistral→Sarvam | mistralai/Mistral-Small-24B-Base-2501 | sarvamai/sarvam-m | 24B | **Indian** SFT+RLVR (GRPO) |

(Gemma access was granted mid-run; Sarvam-M is the Indian fine-tune of the Mistral base — our
"native-alignment" arm, plan §11.5.)

## Compute (and why two clouds)

- **8–9B pairs (fp16): AWS `g6e.xlarge` (1×L40S 48 GB).** One instance scored Llama base/Instruct +
  Gemma base/it sequentially (`scripts/scoring/ec2_run_suite.sh`), ~30–45 min, then terminated.
- **24B pair (fp16): CMU Babel, 4×A6000 (192 GB).** AWS couldn't do 24B fp16 cheaply (L40S 48 GB
  too small; A100-80GB needs a quota increase that routed to a **manual support case 178026522000148**).
  Babel sidesteps the quota: SLURM job on `general` partition, `device_map="auto"` shards the 24B
  across 4×A6000 in fp16. Job `scripts/scoring/babel_suite.sbatch`, id **8226283**.

### Babel specifics (reusable for the 60-cell scale)
- Login: `ssh babel` (`login.babel.cs.cmu.edu`, user anshulk, key id_ed25519, ProxyJump for `babel-*`).
- Account **gneubig**; QOS **normal**; partition **general** (2-day max). The **array** partition needs
  `array_qos` which this account lacks → use `general`.
- 4×A6000 requested (`--gres=gpu:A6000:4 --nodes=1`); A100-80GB nodes were draining.
- Model weights (~48 GB each) go to **node-local `/data/$USER`** scratch (home is 100 GB / ~46 GB free);
  the job auto-discovers it. Results (tiny) written to shared `~/iccd`.
- Base conda has torch 2.10 / transformers 5.0 → the scorer tries `dtype=` then falls back to
  `torch_dtype=` (renamed in tf 5).

## Scorer
`scripts/scoring/stage8_score.py` (single file, runs anywhere): loads a model fp16, computes delta-L per
item (leading-space, teacher-forced target), writes a resumable JSONL. `device_map="auto"` (multi-GPU
shard), Gemma loaded with `attn_implementation="eager"` (correct logits). `scripts/cross_model.py`
combines the per-model results into the appended scores + prints per-pair summary.

## Results (all 3 pairs, n=100)

| Pair | mean ΔL base | mean ΔL aligned | Δ = base−aligned (mean / median / std) | corr(base,aligned) | base>aligned (>1) | aligned>base (<−1) | ~equal (≤1) |
|---|---|---|---|---|---|---|---|
| **Llama-3.1-8B** | 7.95 | 8.36 | −0.41 / −0.37 / 1.65 | 0.86 | 19 | 35 | 46 |
| **Gemma-2-9B** | 7.47 | 9.71 | −2.23 / −2.23 / 2.11 | 0.91 | 6 | 67 | 27 |
| **Mistral→Sarvam-M** | 7.41 | **6.05** | **+1.36 / +0.96 / 3.19** | **0.36** | **49** | 23 | 28 |

**Reading (Axis A = benign regional festivals).** The two **clean within-recipe RLHF** pairs (each
aligned model vs its *own* base) show **no Axis-A selectivity** here — base and aligned are highly
correlated (0.86–0.91), and the aligned models are if anything *more* confident (Δ negative; Gemma
sharpens markedly). On benign cultural facts, instruction-tuning **preserves (Llama) or sharpens
(Gemma)** the representation. This **absence of selectivity is the expected Axis-A baseline, not
evidence against selectivity**: Axis A (festivals) is benign, and selectivity is predicted to bite on
**Axis C sensitive** content, which this single cell does not contain.

The **Indian** alignment (**Sarvam-M**) breaks that pattern: Δ is **positive** (base binds *more
strongly* than the aligned model) and the base↔aligned correlation **drops to 0.36** (vs 0.86–0.91).
This is the only **rewrite-direction** signal in the table — but it sits in the **confounded** arm.
Sarvam differs from the RLHF pairs in **team, training-language mix, and algorithm jointly** (heavy
Indic SFT+RLVR/GRPO; project-plan §3), so the within-pair Δ **cannot be attributed to alignment-recipe
vs language vs team**. It therefore **illustrates the cross-recipe selectivity question; it is not
selectivity evidence and it is not rewrite-confirmed** — a single-cell directional signal in a
confounded arm, nothing more. **Tokenizer-equality gate (plan §8.6) PASSED** — Mistral-base and Sarvam
share vocab 131,072 with **0/21 target mismatches**, so the Δ is a real (not tokenizer-artifact)
within-pair shift; that rules out one artifact, it does **not** turn the confounded signal into a
mechanism verdict. (Sarvam's tokenizer emits a `fix_mistral_regex` warning — a pre-tokenization edge
case that does **not** alter our clean state-name targets; both models scored identically, so the
within-pair Δ holds.)

### Binding-sign cross-tab (does alignment flip the answer on/off?)
Using delta-L sign as a proxy for "the festival still points to the correct state" (**not** true 2AFC
accuracy — see below):

| Pair | both have it | base-only (aligned **lost**) | aligned-only (**gained**) | neither |
|---|---|---|---|---|
| Llama-3.1-8B | 100 | 0 | 0 | 0 |
| Gemma-2-9B | 99 | 0 | 0 | 1 |
| Mistral→Sarvam-M | 99 | 1 | 0 | 0 |

At a **confident** bar (delta-L > 1 nat): Llama 100 both; Gemma 98 both / 1 strengthened / 1 neither;
**Sarvam 95 both / 5 weakened / 0 strengthened.** So within this one cell alignment almost never flips
these facts on/off and the action is in *strength*, with **Sarvam the only arm that modulates
downward** — descriptively consistent with its low correlation. This is a sign-proxy observation, not
a mechanism verdict: whether retained-but-weakened knowledge reflects a **late gate** or a **mid-layer
rewrite** is exactly what a single delta-L scalar **cannot** decide. Recoverability is what separates
them — and the Recovery Test *affirms gating only* (a non-recovery is unresolved, never "rewrite"), so
this is deferred to that test below, not read off the scalar.

**Caveat — this is not accuracy.** delta-L sign means "the festival raises the correct state vs the
corrupted twin," a proxy for *knows-it*, not "outputs the correct state as top-1." True 2-alternative
accuracy needs the **`r'` re-score** (compare `logit(r)` vs `logit(r')` under the clean prefix) — the
behavioral readout specified in plan §10.7, not yet computed.

## What turns this preview into a verdict (forward pointer)

This delta-L preview is a scalar; a scalar **cannot separate a recoverable late gate from an
unrecoverable mid-layer rewrite**. Two things are needed before any mechanism claim, and neither is in
the current pipeline:

- **The Recovery Test (project-plan §7, Phase 4)** — the causal adjudicator. It triangulates
  within-family interchange / cross-checkpoint patching, refusal-direction ablation (+ addition) and
  latent-feature **steering**, and **logit-lens presence** (is the target present mid-stack though
  absent at the output?). Its rule is **gate-affirmative only**: recoverability by a late-locus
  intervention **affirms gating**; **non-recovery is UNRESOLVED, never erasure** (a masked/failed patch
  ≠ absent). This is what would adjudicate the retained-but-weakened Sarvam pattern, which delta-L
  cannot.
- **The 2AFC accuracy behavioral bridge `LD(r,r')`** — `1[ logit(r|clean) > logit(r'|clean) ]`,
  reported as the axis-level interaction in the base→aligned accuracy delta. It requires scoring
  **`logP(r' | clean prefix)`**, which the current delta-L pipeline does **not** capture (it scores r
  under clean vs corrupted, not r' under clean) — the deferred §10.7 scorer addition. Diagnostic on
  the fixed cells, **not** a coverage benchmark; reported alongside, never in place of, the
  representational analysis.

## Status
- **All 3 pairs done and appended** to `data/final/iccd_A01-SS-01.json`. AWS instances terminated;
  **Babel wiped** (`~/iccd` removed, token file removed, node-local scratch auto-purges) — nothing left.
- Tokenizer-equality gate passed for the 24B pair (the one pair where it mattered).
- **Open item:** true 2AFC accuracy (`LD(r,r')`) is the planned §10.7 readout; needs a ~10-line scorer
  addition (also score `r'` under the clean prefix) + one re-score pass to backfill A01-SS-01.
- AWS quota increase (`L-DB2E81BA` → 48 vCPU) still under manual review — only needed if we later
  prefer AWS A100 over Babel for the 24B.
