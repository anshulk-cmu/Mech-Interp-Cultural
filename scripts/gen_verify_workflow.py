"""Generate a Claude Tier-1.5 verification Workflow script for a filtered cell.

Reads data/interim/pairs_filtered_<cell>.json and emits a self-contained workflow .js
that batches the items and has one agent per batch apply the four construct-validity
checks (fact_ok, leakage_ok, counterfactual_ok, natural_ok) with web fact-checking,
returning a structured verdict per item. Workflow scripts cannot read disk, so the
items are embedded into the generated script.

Usage:  python scripts/gen_verify_workflow.py A01-EE-01 [batch_size]
Output: data/interim/verify_<cell>.workflow.js  (then run via Workflow{scriptPath:...})
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

INTERIM = Path("data/interim")

# Per-sub-concept Tier-1.5 verification profile: what the anchor must BE for fact_ok, and the
# binding phrase. Keyed by the cell's sub-concept code ("01" festivals, "02" costume & textile).
# Leakage / counterfactual / naturalness checks are shared; only fact_ok and the framing differ.
_VERIFY_PROFILE = {
    "01": {
        "item": "festival",
        "binding": "binds a specific Indian festival to its specific Indian STATE",
        "fact_ok": ("Is the anchor a REAL, traditional festival / fair / mela / jatra / utsav / urs "
                    "genuinely and distinctively celebrated in the target state? Use the provided "
                    "wiki_extract first; web-search to confirm when the extract is empty or insufficient. "
                    "fact_ok = FALSE if: it is NOT a festival (it is a temple, a monument, a person, an "
                    "organization/association, a place, or a film/music/literary/poetry/sports/food/flower/"
                    "TOURISM-promotional event or a modern \"<City> Mahotsav/Day\" branding); OR it is "
                    "actually associated with a DIFFERENT Indian state; OR you cannot verify it from any "
                    "real source."),
    },
    "02": {
        "item": "textile / costume",
        "binding": "binds a specific Indian textile, weave, or regional garment to its specific Indian STATE",
        "fact_ok": ("Is the anchor a REAL, distinctive regional TEXTILE / weave / handloom / saree / "
                    "embroidery / regional garment genuinely and distinctively associated with the target "
                    "state (a recognized regional craft, ideally GI-tagged or documented as that state's "
                    "tradition)? Use the provided wiki_extract first; web-search (Wikipedia, the GI "
                    "registry, state handloom/handicraft boards) to confirm when the extract is empty or "
                    "insufficient. fact_ok = FALSE if: it is NOT a textile/weave/garment/embroidery (it is "
                    "a festival, a food/dish, a dance, a person, an organization, a museum, or a bare "
                    "place/city/district/region NAME on its own); OR it is a pan-Indian, non-distinctive "
                    "garment with no regional identity (a plain 'saree', 'dhoti', 'kurta', 'lungi', "
                    "'turban' etc.); OR it is actually a craft of a DIFFERENT Indian state; OR you cannot "
                    "verify it from any real source. A textile NAMED AFTER a town in the target state "
                    "(e.g. Banarasi, Sambalpuri, Chanderi, Pochampally) is the normal naming convention "
                    "and PASSES fact_ok — judge leakage separately in check 2."),
    },
}


def build(cid: str, batch: int = 10) -> Path:
    pairs = json.load(open(INTERIM / f"pairs_filtered_{cid}.json", encoding="utf-8"))
    items = [{
        "item_id": p["item_id"],
        "anchor": p["anchor"],
        "target": p["target"],
        "clean_prefix": p["clean_prefix"],
        "corrupted_prefix": p["corrupted_prefix"],
        "corruptor_anchor": p["corruptor_anchor"],
        "r_prime": p["r_prime"],
        "wiki_extract": (p.get("wiki", {}) or {}).get("extract", "")[:280].replace("`", "'").replace("${", "$ {"),
        "source_primary": p.get("source_primary"),
        "web_source_url": p.get("web_source_url"),
    } for p in pairs]
    batches = [items[i:i + batch] for i in range(0, len(items), batch)]

    js = '''export const meta = {
  name: 'iccd-verify-%CID%',
  description: 'Tier-1.5 construct-validity verification for %CID% (fact/leakage/counterfactual/naturalness)',
  phases: [{ title: 'Verify', detail: 'fact-check each anchor->state minimal pair' }],
}

const BATCHES = %BATCHES%

const SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    verdicts: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: {
        item_id: { type: 'string' },
        fact_ok: { type: 'boolean' },
        leakage_ok: { type: 'boolean' },
        counterfactual_ok: { type: 'boolean' },
        natural_ok: { type: 'boolean' },
        pass: { type: 'boolean' },
        reason: { type: 'string' },
        source: { type: 'string' },
      },
      required: ['item_id', 'fact_ok', 'leakage_ok', 'counterfactual_ok', 'natural_ok', 'pass', 'reason', 'source'],
    } },
  },
  required: ['verdicts'],
}

function promptFor(batch) {
  return `You are doing construct-validity verification for a controlled minimal-pair research dataset. Axis A tests whether a model %BINDING%. Each item has a clean prefix naming a %ITEM% (the anchor) whose correct completion is the target state, and a corrupted prefix naming a DIFFERENT-region %ITEM% (the corruptor).

>>> CRITICAL OUTPUT RULE: your FINAL action MUST be a call to the StructuredOutput tool with one verdict per item_id. Do NOT end your turn with a prose summary — an answer with no tool call is a total failure. These items are ALREADY web-sourced (each carries a source + wiki_extract), so JUDGE construct validity from the provided evidence + your own knowledge. Only web-search the FEW items that genuinely look doubtful or unfamiliar; do NOT exhaustively re-search every item. Budget at most ~5 web searches for the whole batch, then immediately CALL StructuredOutput. <<<

For EACH item below, apply these four checks (each a boolean) and decide pass = (all four true):

1. fact_ok — %FACT_OK% Use the provided evidence + your knowledge first; web-search ONLY if genuinely unsure. Do not manufacture facts.

2. leakage_ok — Does the clean prefix avoid leaking the answer? FALSE if the anchor text contains the target state's own name, OR a major city/district/region name that gives the state away (e.g. Lucknow/Prayagraj/Varanasi->UP, Jaipur/Jodhpur/Udaipur->Rajasthan, Amritsar/Ludhiana->Punjab, Dehradun/Haridwar->Uttarakhand, Leh->Ladakh, Kolkata->West Bengal, Patna/Gaya->Bihar, Guwahati->Assam, Bhubaneswar/Cuttack/Puri->Odisha, Imphal->Manipur, Shillong->Meghalaya, Aizawl->Mizoram, Kohima->Nagaland, Agartala->Tripura, Gangtok->Sikkim). NOTE: a craft NAMED AFTER its town of origin (Banarasi, Sambalpuri, Chanderi, Pochampally, Kanchipuram, Mangalagiri) does NOT contain the STATE name and PASSES leakage_ok; only flag FALSE when the explicit state name or an unmistakable city giveaway is literally present in the anchor.

3. counterfactual_ok — Is corruptor_anchor a REAL %ITEM% from a genuinely DIFFERENT region than the target (its state r_prime must differ from target), and does it read as a plausible same-template item? FALSE if the corruptor is malformed, not real, or actually from the target's own state.

4. natural_ok — Do BOTH prefixes read as natural English naming a single clean %ITEM%? FALSE for parentheticals, commas, embedded prepositions ("X in Y"), mojibake, truncation fragments, bare language words, or non-name strings.

Be strict but fair: genuine, distinctive regional items should PASS; noise (wrong-category entries, places, orgs, mis-attributions, pan-Indian non-distinctive items, malformed names) should FAIL the relevant check. In 'reason' give a one-line justification; in 'source' cite the URL or wiki page that confirms fact_ok (or "n/a" if failed). Return exactly one verdict per item_id, then CALL StructuredOutput (mandatory final action — do not stop without it).

ITEMS:
${batch.map((it, i) => `${i + 1}. item_id=${it.item_id}
   anchor="${it.anchor}"  target="${it.target}"
   clean="${it.clean_prefix}"
   corrupted="${it.corrupted_prefix}"  corruptor="${it.corruptor_anchor}" (r_prime=${it.r_prime})
   source_primary=${it.source_primary}  web_source=${it.web_source_url || 'none'}
   wiki_extract="${(it.wiki_extract || '').replace(/"/g, "'")}"`).join('\\n')}`
}

const results = await parallel(
  BATCHES.map((b, i) => () =>
    agent(promptFor(b), { label: `verify:%CID%:b${i}`, phase: 'Verify', agentType: 'general-purpose', schema: SCHEMA })
      .then((r) => (r && r.verdicts) || [])
  )
)
const all = []
for (const v of results.filter(Boolean)) all.push(...v)
log(`%CID% verified: ${all.length} verdicts; ${all.filter(v => v.pass).length} pass`)
return { verdicts: all }
'''
    sub = cid.split("-")[-1]
    prof = _VERIFY_PROFILE.get(sub, _VERIFY_PROFILE["01"])
    js = (js.replace("%BATCHES%", json.dumps(batches, ensure_ascii=False))
            .replace("%BINDING%", prof["binding"]).replace("%ITEM%", prof["item"])
            .replace("%FACT_OK%", prof["fact_ok"]).replace("%CID%", cid))
    out = INTERIM / f"verify_{cid}.workflow.js"
    out.write_text(js, encoding="utf-8")
    print(f"{cid}: {len(items)} items -> {len(batches)} batches -> {out}")
    return out


if __name__ == "__main__":
    cid = sys.argv[1]
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    build(cid, bs)
