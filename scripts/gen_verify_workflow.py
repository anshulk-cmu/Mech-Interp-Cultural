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
  phases: [{ title: 'Verify', detail: 'fact-check each festival->state minimal pair' }],
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
  return `You are doing construct-validity verification for a controlled minimal-pair research dataset. Axis A tests whether a model binds a specific Indian festival to its specific Indian STATE. Each item has a clean prefix naming a festival (the anchor) whose correct completion is the target state, and a corrupted prefix naming a DIFFERENT-region festival (the corruptor).

For EACH item below, apply these four checks (each a boolean) and decide pass = (all four true):

1. fact_ok — Is the anchor a REAL, traditional festival / fair / mela / jatra / utsav / urs genuinely and distinctively celebrated in the target state? Use the provided wiki_extract first; web-search to confirm when the extract is empty or insufficient. fact_ok = FALSE if: it is NOT a festival (it is a temple, a monument, a person, an organization/association, a place, or a film/music/literary/poetry/sports/food/flower/TOURISM-promotional event or a modern "<City> Mahotsav/Day" branding); OR it is actually associated with a DIFFERENT Indian state; OR you cannot verify it from any real source. NEVER assume — verify. Do not manufacture facts.

2. leakage_ok — Does the clean prefix avoid leaking the answer? FALSE if the anchor text contains the target state's own name, OR a major city/district/region name that gives the state away (e.g. Lucknow/Prayagraj/Varanasi->UP, Jaipur/Jodhpur/Udaipur->Rajasthan, Amritsar/Ludhiana->Punjab, Dehradun/Haridwar->Uttarakhand, Leh->Ladakh, Kolkata->West Bengal, Patna/Gaya->Bihar, Guwahati->Assam, Bhubaneswar/Cuttack/Puri->Odisha, Imphal->Manipur, Shillong->Meghalaya, Aizawl->Mizoram, Kohima->Nagaland, Agartala->Tripura, Gangtok->Sikkim).

3. counterfactual_ok — Is corruptor_anchor a REAL festival from a genuinely DIFFERENT region than the target (its state r_prime must differ from target), and does it read as a plausible same-template festival? FALSE if the corruptor is malformed, not a real festival, or actually from the target's own state.

4. natural_ok — Do BOTH prefixes read as natural English naming a single clean festival? FALSE for parentheticals, commas, embedded prepositions ("X in Y"), mojibake, truncation fragments, bare language words, or non-name strings.

Be strict but fair: genuine, distinctive regional festivals should PASS; noise (events, temples, orgs, mis-attributions, pan-Indian non-distinctive festivals, malformed names) should FAIL the relevant check. In 'reason' give a one-line justification; in 'source' cite the URL or wiki page that confirms fact_ok (or "n/a" if failed). Return exactly one verdict per item_id. Then CALL StructuredOutput.

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
    js = js.replace("%CID%", cid).replace("%BATCHES%", json.dumps(batches, ensure_ascii=False))
    out = INTERIM / f"verify_{cid}.workflow.js"
    out.write_text(js, encoding="utf-8")
    print(f"{cid}: {len(items)} items -> {len(batches)} batches -> {out}")
    return out


if __name__ == "__main__":
    cid = sys.argv[1]
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    build(cid, bs)
