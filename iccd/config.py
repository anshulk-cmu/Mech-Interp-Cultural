"""Locked design constants for ICCD-6K Phase 1.

Every value here traces to plans/ICCD-6K-plan.md (the validated design source).
Breadth is FIXED: 60 cells = 3 axes x 5 regions x 4 sub-concepts. The smoke test
exercises one cell (A01-SS-01) but the full design is encoded so scaling is a config flip.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Roots ---------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
FINAL = DATA / "final"
RESOURCES = DATA / "resources"
LOGS = REPO_ROOT / "logs"
for _d in (RAW, INTERIM, FINAL, RESOURCES, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

# --- Reference model / tokenizer (plan: user-corrected to 3.1) -----------
MODEL_REPO = "meta-llama/Llama-3.1-8B"      # official, gated; token has access
TOKENIZER_REPO = MODEL_REPO                  # same Llama-3 tokenizer as 3.0

# --- Locked seeds (plan Section 3.8 / 10.1) ------------------------------
SEEDS = {"main": 42, "holdout": 137, "spot_check": 314, "counterfactual_audit": 271}

# --- Axes (plan Section 3.1) ---------------------------------------------
AXES = {
    "A01": "Regional Specificity",
    "A02": "Cultural Flattening",
    "A03": "Sensitive Policy",
}

# --- Sub-concepts, 4 per axis (plan Section 3.3) -------------------------
SUBCONCEPTS = {
    "A01": {"01": "Festivals", "02": "Costume and Textile", "03": "Cuisine", "04": "Rituals and Ceremonies"},
    "A02": {"01": "Classical Dance", "02": "Classical Music", "03": "Visual Art", "04": "Architecture and Built Form"},
    "A03": {"01": "Social Structure and Caste", "02": "Religion and Scripture", "03": "History and Political Memory", "04": "Traditional Medicine"},
}

# --- Five-region taxonomy, locked & frozen (plan Section 3.2) ------------
REGIONS = {
    "NN": ["Punjab", "Haryana", "Himachal Pradesh", "Jammu and Kashmir", "Ladakh",
           "Rajasthan", "Uttar Pradesh", "Uttarakhand", "Delhi", "Chandigarh"],
    "SS": ["Andhra Pradesh", "Karnataka", "Kerala", "Tamil Nadu", "Telangana",
           "Puducherry", "Lakshadweep", "Andaman and Nicobar Islands"],
    "EE": ["Bihar", "Jharkhand", "Odisha", "West Bengal", "Sikkim", "Arunachal Pradesh",
           "Assam", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Tripura"],
    "WW": ["Gujarat", "Maharashtra", "Goa", "Dadra and Nagar Haveli and Daman and Diu"],
    "CC": ["Madhya Pradesh", "Chhattisgarh"],
}
REGION_NAMES = {"NN": "North", "SS": "South", "EE": "East", "WW": "West", "CC": "Central"}
STATE_TO_REGION = {state: rc for rc, states in REGIONS.items() for state in states}

# --- Axis-A STR relation templates, per sub-concept (plan Section 3.7 / App. D) ---
# Cross-anchor-swap STR: within a pair ONLY the anchor token changes, so the relation
# frame is one fixed string per cell. A01-01 reproduces the released festival template
# verbatim; A01-02 is the plan §D.2 textile frame. 03/04 are provisional defaults
# (finalize when those cells are built). Keyed by (axis, sub_concept).
RELATION_TEMPLATES = {
    ("A01", "01"): "{anchor} is a festival celebrated in the Indian state of",
    ("A01", "02"): "{anchor} is a folk textile tradition associated with the Indian state of",
    ("A01", "03"): "{anchor} is a traditional dish particularly associated with the Indian state of",
    ("A01", "04"): "{anchor} is a traditional ritual observed in the Indian state of",
}


def relation_template(axis: str, sub: str) -> str:
    """The fixed STR relation frame for a cell (KeyError if not yet defined)."""
    return RELATION_TEMPLATES[(axis, sub)]


# --- Cell design (plan Section 3.4) --------------------------------------
PER_CELL_FINAL = 100
CANDIDATES_PER_CELL = 240            # over-sampling buffer (plan Section 3.8)
PASS_A_TARGET = 143                  # post-Stage-4 survivors expected (240 * 0.60)
# Target token-length balance per 100 final items. Spans 1-5 tokens since MAX_TARGET_TOKENS
# went 3->5 (user-authorized permanent change 2026-06-05). The split is a TARGET; the realized
# mix is constrained by which states a cell actually has and is recorded. Stage-5 fallback fill
# covers strata a cell cannot populate (e.g. NN/SS have no 5-token eligible state).
TOKEN_BALANCE = {1: 30, 2: 25, 3: 20, 4: 15, 5: 10}  # sums to 100
MAX_TARGET_TOKENS = 5               # permanent increase 3->5 (user-authorized 2026-06-05)
# With the 5-token cap every previously hard-excluded state EXCEPT 6+-token names (Andaman and
# Nicobar Islands=6, Dadra and Nagar Haveli and Daman and Diu=13) is now natively eligible, so no
# per-state F1 exception is needed (Chhattisgarh=5 now passes on its own). Kept as an (empty) hook.
F1_TOKEN_EXCEPTIONS: set[str] = set()
FLOOR_PRIMARY = 50

# --- Stage-4 filter bounds (plan Section 6.4) ----------------------------
PREFIX_MIN_TOKENS = 8
PREFIX_MAX_TOKENS = 64
SUFFIX_MATCH_TOKENS = 4
# generic nouns allowed to survive corruption (F4 blocklist, plan Section 6.4)
GENERIC_NOUN_BLOCKLIST = {"sari", "dance", "festival", "music", "cuisine", "fair",
                          "print", "mela", "ikat", "dish", "harvest",
                          # Cuisine (A01-03) generic food nouns
                          "curry", "masala", "sweet", "snack", "bread", "rice", "gravy",
                          "fry", "pickle", "chutney", "thali", "dal", "sabzi"}

# --- Stage-8 retention threshold (plan Section 8.3) ----------------------
DELTA_L_FLOOR_NATS = 1.0

# --- SANSKRITI ingestion (plan Section 5.1 / 6.1; map validated vs CSV 2026-05-31) ---
SANSKRITI_REPO = "13ari/Sanskriti"
SANSKRITI_FILE = "Merged_Dataset_english_SANSKRITI.csv"
SANSKRITI_KEEP_QTYPES = {"Association", "State Prediction"}

# SANSKRITI uses underscored state names; map to our canonical 5-region names.
SANSKRITI_STATE_FIX = {
    "Jammu_kashmir": "Jammu and Kashmir",
    "Andaman_and_Nicobar": "Andaman and Nicobar Islands",
    "Dadra_and_Nagar_Haveli_and_Daman_and_Diu": "Dadra and Nagar Haveli and Daman and Diu",
}

# SANSKRITI attribute -> (axis, subconcept). Only clean 1:1 maps; others dropped.
# Gaps (full build, not smoke): 'Dance_and_Music' lumps B01+B02; no attribute for
# B04 Architecture or C01 Caste -> those cells lean on Wikipedia/web (plan Section 5.1).
ATTRIBUTE_MAP = {
    "Festivals": ("A01", "01"),
    "Costume": ("A01", "02"),
    "Cuisine": ("A01", "03"),
    "Rituals_and_Ceremonies": ("A01", "04"),
    "Art": ("A02", "03"),
    "Religion": ("A03", "02"),
    "History": ("A03", "03"),
    "Medicine": ("A03", "04"),
}


def normalize_sanskriti_state(s: str):
    """SANSKRITI state string -> canonical state name, or None if out of scope."""
    s = str(s).strip()
    name = SANSKRITI_STATE_FIX.get(s, s.replace("_", " "))
    return name if name in STATE_TO_REGION else None

# --- Wikipedia client (plan Section 5.3) ---------------------------------
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_USER_AGENT = "ICCD-Research/1.0 (contact: anshulk@andrew.cmu.edu)"
WIKI_SNAPSHOT_DATE = "2026-02-01"   # pinned target; actual oldid recorded per page

# --- Reusable curation assets (scale to all 60 cells) ---
CORRUPTOR_BANK = RESOURCES / "corruptor_bank.json"   # clean cross-region corruptor festivals
WEB_FESTIVALS = RESOURCES / "web_festivals.json"     # per-cell web-tier supplement
WEB_SOURCES_LOG = RESOURCES / "web_sources.jsonl"    # provenance log for web-tier items


def cell_id(axis: str, region: str, subconcept: str) -> str:
    """Six-token cell code, e.g. A01-SS-01."""
    return f"{axis}-{region}-{subconcept}"


def cell_label(axis: str, region: str, subconcept: str) -> str:
    return f"{AXES[axis]} / {REGION_NAMES[region]} / {SUBCONCEPTS[axis][subconcept]}"


def get_hf_token() -> str | None:
    """HF token from env, else the Anthropic_Fellows_Research .env (kept OUT of this repo)."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    env_path = Path(r"D:\Anthropic_Fellows_Research\.env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("hf_"):
                return line.split()[0]
    return None


SMOKE_CELL = ("A01", "SS", "01")
