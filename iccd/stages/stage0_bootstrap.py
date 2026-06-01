"""Stage 0 — bootstrap leakage resources: gazetteer (F2), language_map (F3), tokenizer pin.

Gazetteer = district/city names per state for geographic-leakage detection. The
language map is hand-curated and frozen so F3 is deterministic. Smoke builds the
South region; the full region set is a config flip.
"""
from __future__ import annotations

import sys

from huggingface_hub import HfApi

from .. import wiki
from ..config import REGIONS, RESOURCES, TOKENIZER_REPO, get_hf_token
from ..logutil import get_logger
from ..state import write_json

log = get_logger("stage0")
GAZ = RESOURCES / "gazetteer.json"
LANG = RESOURCES / "language_map.json"
TOKPIN = RESOURCES / "tokenizers.json"

_CAT_TEMPLATES = ["Districts of {s}", "Cities and towns in {s}", "Districts of {s}, India"]

LANGUAGE_MAP = {
    # South
    "Andhra Pradesh": ["Telugu"], "Telangana": ["Telugu"], "Karnataka": ["Kannada"],
    "Kerala": ["Malayalam"], "Tamil Nadu": ["Tamil"],
    "Puducherry": ["Tamil", "French", "Telugu", "Malayalam"],
    "Lakshadweep": ["Malayalam"], "Andaman and Nicobar Islands": ["Hindi", "Bengali", "Tamil"],
    # North
    "Punjab": ["Punjabi"], "Haryana": ["Haryanvi", "Hindi"],
    "Himachal Pradesh": ["Pahari", "Hindi"],
    "Jammu and Kashmir": ["Kashmiri", "Dogri", "Urdu"], "Ladakh": ["Ladakhi"],
    "Rajasthan": ["Rajasthani", "Marwari", "Hindi"],
    "Uttar Pradesh": ["Hindi", "Awadhi", "Braj", "Bhojpuri", "Urdu"],
    "Uttarakhand": ["Garhwali", "Kumaoni", "Hindi"],
    "Delhi": ["Hindi", "Punjabi", "Urdu"], "Chandigarh": ["Hindi", "Punjabi"],
    # East
    "Bihar": ["Maithili", "Bhojpuri", "Magahi", "Hindi"],
    "Jharkhand": ["Santali", "Nagpuri", "Hindi"], "Odisha": ["Odia"],
    "West Bengal": ["Bengali"], "Sikkim": ["Nepali", "Bhutia", "Lepcha"],
    "Arunachal Pradesh": ["Nyishi", "Adi", "Apatani", "Monpa"],
    "Assam": ["Assamese", "Bodo"], "Manipur": ["Meitei", "Manipuri"],
    "Meghalaya": ["Khasi", "Garo"], "Mizoram": ["Mizo"],
    "Nagaland": ["Nagamese", "Ao", "Angami"], "Tripura": ["Kokborok", "Bengali"],
    # West
    "Gujarat": ["Gujarati"], "Maharashtra": ["Marathi"], "Goa": ["Konkani", "Marathi"],
    "Dadra and Nagar Haveli and Daman and Diu": ["Gujarati", "Marathi", "Hindi"],
    # Central
    "Madhya Pradesh": ["Hindi"], "Chhattisgarh": ["Chhattisgarhi", "Hindi"],
}


def _strip(title: str) -> str:
    return title.split(" (")[0].strip()


def build_gazetteer(states):
    gaz = {}
    for st in states:
        places = set()
        for tmpl in _CAT_TEMPLATES:
            for t in wiki.category_members(tmpl.format(s=st)):
                places.add(_strip(t))
        gaz[st] = sorted(places)
        log.info("  gazetteer %s: %d places", st, len(gaz[st]))
    return gaz


def run(states=None, force: bool = False):
    states = states or REGIONS["SS"]
    if GAZ.exists() and LANG.exists() and TOKPIN.exists() and not force:
        log.info("outputs present, skipping (force=True to rebuild)")
        return
    write_json(GAZ, build_gazetteer(states))
    write_json(LANG, {s: LANGUAGE_MAP.get(s, []) for s in states})
    sha = HfApi().model_info(TOKENIZER_REPO, token=get_hf_token()).sha
    write_json(TOKPIN, {TOKENIZER_REPO: {"revision": sha, "role": "reference balance + delta-L"}})
    log.info("tokenizer pin %s @ %s", TOKENIZER_REPO, sha)
    log.info("wrote gazetteer (%d states), language_map, tokenizers.json", len(states))


if __name__ == "__main__":
    run(force="--force" in sys.argv)
