"""Dev utility: validate gated tokenizer access + inspect SANSKRITI schema.

Run once before writing Stage 1, so the SANSKRITI->cell mapping is built against
the real columns/attribute names, not a guess. Requires PYTHONPATH=repo root.
"""
from huggingface_hub import hf_hub_download
import pandas as pd

from iccd.config import SANSKRITI_REPO, SANSKRITI_FILE, RAW, get_hf_token
from iccd.tok import get_tokenizer, target_token_len

print("=== tokenizer (validates gated Llama-3.1 access) ===")
tok = get_tokenizer()
print("class:", tok.__class__.__name__, "| vocab:", tok.vocab_size)
for t in ["Tamil Nadu", "Kerala", "Onam", "Pongal", "Andhra Pradesh", "Bihu"]:
    print(f"  ' {t}' -> {target_token_len(t)} target tokens")

print("\n=== download + inspect SANSKRITI ===")
dest = RAW / "sanskriti"
dest.mkdir(parents=True, exist_ok=True)
path = hf_hub_download(repo_id=SANSKRITI_REPO, filename=SANSKRITI_FILE,
                       repo_type="dataset", local_dir=str(dest), token=get_hf_token())
print("downloaded:", path)
df = pd.read_csv(path)
print("rows:", len(df), "| cols:", list(df.columns))
print("\n--- head(3) ---")
print(df.head(3).to_string())
for c in df.columns:
    nun = df[c].nunique(dropna=True)
    if nun <= 45:
        print(f"\n{c} ({nun} unique): {sorted(map(str, df[c].dropna().unique()))}")
    else:
        print(f"\n{c}: {nun} unique (sample): {list(map(str, df[c].dropna().unique()[:8]))}")
