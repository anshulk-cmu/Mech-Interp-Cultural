"""Append per-model Stage-8 scores + cross-model rewrite-vs-gate deltas onto the release.

For each base<->aligned pair: delta = delta-L_base - delta-L_aligned, where
delta-L = logP(target|clean) - logP(target|corrupted) (Zhang & Nanda logit-difference).
delta ~ 0 means alignment left the festival->state binding intact; large positive means the
base binds it more strongly than the aligned model (a candidate gating/rewrite shift).

Writes model_scores + cross_model_delta into data/final/iccd_A01-SS-01.json in place
(idempotent: rebuilt from whatever results_<slug>.jsonl exist, so re-runs add new models),
and prints per-pair summary stats. Pairs with no results yet are skipped.
"""
import io
import json
import statistics as st
import sys
from pathlib import Path

INTERIM = Path("data/interim")
# slug -> (display key in JSON, pair name, role)
MODELS = {
    "llama31-base": ("Llama-3.1-8B-base", "Llama-3.1-8B", "base"),
    "llama31-inst": ("Llama-3.1-8B-Instruct", "Llama-3.1-8B", "aligned"),
    "gemma2-base": ("gemma-2-9b-base", "Gemma-2-9B", "base"),
    "gemma2-it": ("gemma-2-9b-it", "Gemma-2-9B", "aligned"),
    "mistral-base": ("Mistral-Small-24B-Base", "Mistral->Sarvam-M", "base"),
    "sarvam-m": ("Sarvam-M", "Mistral->Sarvam-M", "aligned"),
}


def load(slug):
    p = INTERIM / f"results_{slug}.jsonl"
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["item_id"]] = {"delta_L": round(r["delta_L"], 4),
                                     "logp_clean": round(r["logp_clean"], 4),
                                     "logp_corrupt": round(r["logp_corrupt"], 4)}
    return out


def _corr(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else float("nan")


def main(cell="A01-SS-01"):
    RELEASE = Path(f"data/final/iccd_{cell}.json")
    rel = json.load(io.open(RELEASE, encoding="utf-8"))
    scores = {slug: load(slug) for slug in MODELS}

    pairs = {}  # pair name -> (base_slug, aligned_slug)
    for slug, (_, pair, role) in MODELS.items():
        pairs.setdefault(pair, {})[role] = slug

    for item in rel:
        iid = item["item_id"]
        ms = {MODELS[s][0]: scores[s][iid] for s in MODELS if iid in scores[s]}
        item["model_scores"] = ms
        cmd = {}
        for pair, rs in pairs.items():
            b, a = rs.get("base"), rs.get("aligned")
            if b and a and iid in scores[b] and iid in scores[a]:
                cmd[pair] = round(scores[b][iid]["delta_L"] - scores[a][iid]["delta_L"], 4)
        item["cross_model_delta"] = cmd

    json.dump(rel, io.open(RELEASE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    for pair, rs in pairs.items():
        b, a = rs.get("base"), rs.get("aligned")
        common = [it["item_id"] for it in rel
                  if b and a and it["item_id"] in scores[b] and it["item_id"] in scores[a]]
        if not common:
            print(f"\n=== {pair} ===  (pending)")
            continue
        base = [scores[b][i]["delta_L"] for i in common]
        algn = [scores[a][i]["delta_L"] for i in common]
        delta = [x - y for x, y in zip(base, algn)]
        print(f"\n=== {pair}  (n={len(common)}) ===")
        print(f"  mean delta-L  base={st.mean(base):6.2f}   aligned={st.mean(algn):6.2f}")
        print(f"  delta (base - aligned):  mean={st.mean(delta):+.2f}  median={st.median(delta):+.2f}  std={st.pstdev(delta):.2f}")
        print(f"  corr(base,aligned)={_corr(base, algn):.3f}   "
              f"base>aligned(>1):{sum(d > 1 for d in delta)}  aligned>base(<-1):{sum(d < -1 for d in delta)}  "
              f"~equal(|d|<=1):{sum(abs(d) <= 1 for d in delta)}")
    print(f"\nappended model_scores + cross_model_delta to {RELEASE}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "A01-SS-01")
