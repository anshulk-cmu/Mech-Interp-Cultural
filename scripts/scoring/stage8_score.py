"""Stage 8 GPU scorer (standalone; runs on the AWS g6.xlarge, scp'd over).

Reads the Pass-A batch, loads Llama-3.1-8B in fp16, computes per-item delta-L in nats
(log-odds, never probability) on the fixed target with a leading space, teacher-forced:
    delta-L = logP(target | clean_prefix) - logP(target | corrupted_prefix)
Retains Axis-A items with delta-L > threshold (clean must beat corrupted). Writes a
resumable JSONL so an interrupted run continues. No iccd imports -> single-file copy.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def sequence_log_prob(model, tok, prefix, target, device):
    pre = tok.encode(prefix)
    tgt = tok.encode(" " + target, add_special_tokens=False)
    ids = torch.tensor([pre + tgt], device=device)
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    lp = torch.log_softmax(logits, dim=-1)
    return sum(lp[len(pre) + i - 1, t].item() for i, t in enumerate(tgt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--threshold", type=float, default=1.0)
    a = ap.parse_args()

    items = json.load(open(a.input))
    out = Path(a.output)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["item_id"])

    token = os.environ.get("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(a.model, token=token)
    # device_map="auto" streams weights to GPU(s) and shards a 24B across multiple GPUs in fp16
    # (g6e.12xlarge = 4xL40S); low_cpu_mem_usage avoids a CPU-first OOM on small-RAM hosts.
    kw = dict(device_map="auto", low_cpu_mem_usage=True, token=token)
    if "gemma" in a.model.lower():
        kw["attn_implementation"] = "eager"  # Gemma-2 logits are correct only with eager attention
        # Gemma-2's sliding-window KV-cache update crashes when sharded across GPUs
        # (RuntimeError: indices ... same device). The 9B fits one L4/L40S, so pin it to a single
        # device — safe on 1-GPU boxes too (device_map="auto" already lands there).
        kw["device_map"] = {"": 0}
    try:  # transformers >=5 renamed torch_dtype -> dtype
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, **kw)
    model.eval()
    print("loaded", a.model, "| tokenizer vocab", tok.vocab_size)

    with out.open("a") as f:
        kept = scored = 0
        for it in items:
            if it["item_id"] in done:
                continue
            lc = sequence_log_prob(model, tok, it["clean_prefix"], it["target"], "cuda")
            lk = sequence_log_prob(model, tok, it["corrupted_prefix"], it["target"], "cuda")
            dl = lc - lk
            keep = dl > a.threshold
            f.write(json.dumps({"item_id": it["item_id"], "delta_L": dl,
                                "logp_clean": lc, "logp_corrupt": lk, "kept": keep}) + "\n")
            f.flush()
            scored += 1
            kept += int(keep)
        print(f"scored {scored} new ({len(done)} cached), kept {kept} this run")


if __name__ == "__main__":
    main()
