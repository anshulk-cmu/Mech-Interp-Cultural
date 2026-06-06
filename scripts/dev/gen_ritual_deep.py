"""Generate the Batch-B deep/top-up ritual research Workflow (WW+CC maximize, EE thin-state top-up).
Embeds held-anchor EXCLUDE lists from held_eewwcc04.json + fresh long-tail seeds. Run:
  python scripts/dev/gen_ritual_deep.py
"""
import json
from pathlib import Path

OUT = Path("data/interim")
held = json.load(open(OUT / "held_eewwcc04.json", encoding="utf-8"))

# fresh long-tail seeds (NEW, excluding held)
SEED = {
 ("A01-WW-04", "Gujarat"): "Meldi Mata rites, Hadkai Mata rites, Sikotar Mata worship, Verai Mata rites, Mahakali Pavagadh rites, Annakut Haveli rite, Hindola darshan, Bhuva possession, Dakla rite, Gatrad Mata, Chamunda rites, Shamlaji rites, Ramdev Pir Gujarat, Dhaja chadhavi -- Gujarati goddess/Bhuva/Haveli rites",
 ("A01-WW-04", "Maharashtra"): "Vetal worship, Bhutoba worship, Janai Devi rites, Tulja Bhavani Shilangan, Renuka Lagna rites, Kshetrapal worship, Mariaai rites, Pat puja, Devrushi possession, Cheda worship, Jakhmata rites, Vagh Baras, Khelya rite, Naivedya Khandoba -- Maharashtrian gramdaivat/possession rites",
 ("A01-WW-04", "Goa"): "Romta Mell, Ghode Modni, Tonyamel, Morlecho Khell, Chorotsav, Gulalotsav, Sheni Uzo, Ranmale, Mussoll, Kala rites, Dhillo, Shigmo Jagor, Maand rite, Kalo -- Goan Shigmo/village-deity rites",
 ("A01-CC-04", "Madhya Pradesh"): "Bhagoria rite, Gowari rite, Nawai rite, Divaso, Babadev worship, Hardaul worship, Dulha Dev rite, Bhuria Baba, Thakur Dev rite, Sheetla MP, Goga Navami MP, Lota Baba, Tejaji MP, Indal -- Madhya Pradesh Bhil/Gond/Nimar deity rites",
 ("A01-CC-04", "Chhattisgarh"): "Kalash Sthapana, Deri Gadhai, Nisha Jatra, Bhitar Raini, Bahar Raini, Kachan Devi rite, Kutumb Jatra, Doli Vidai, Anga Dev rite, Pen Karsad, Mati Pujan, Jeeth Jagar, Danteshwari rites, Maoli Mata rite -- Chhattisgarh Bastar Dussehra ritual stages + Gond rites",
 ("A01-EE-04", "Meghalaya"): "Pomblang Syiem rites, Niam Khasi rite, Mei Ramew worship, Ka Pynbah, Seng Kut Snem, U Ryngkew worship, Iing Sad rite, Wangala rites, Ka Shad Mastieh, Daloi rite -- Khasi/Jaintia/Garo rites",
 ("A01-EE-04", "Mizoram"): "Mim Kut rites, Pawl Kut rites, Thangchhuah ceremony, Inthawi sacrifice, Kawngpui Siam, Lashi worship, Khal sacrifice, Chawngchen, Bawh Pui -- Mizo merit-feast/sacrifice rites",
 ("A01-EE-04", "Assam"): "Suwori, Subachani Puja, Ai Sabah, Nara Singha rite, Tora rite, Borxabah rite, Maroi Puja, Tiloni rite, Doul rite, Manasha Assam -- Assamese/Bodo deity rites",
 ("A01-EE-04", "Bihar"): "Goraiya worship, Brahm Baba worship, Dina Bhadri rite, Manjusha Puja, Sokha Puja, Garhpur Baba, Dak rite, Sokha worship, Loik rite -- Bihari/Mithila folk-deity rites",
 ("A01-EE-04", "Sikkim"): "Lhabab Duchen, Drupka Teshi, Sonam Losar, Namsoong, Kabi Lungtsok rite, Tamu Lochar, Ramfat rite, Bhanu Jayanti rites, Saga Dawa Sikkim -- Sikkimese Bhutia/Lepcha/Limboo rites",
 ("A01-EE-04", "Tripura"): "Tripura Sundari rites, Buisu rites, Sangrai, Wakhrwtwi, Maikatal, Lebang rite, Hangrai snan, Khachi puja, Goria Puja Tripura -- Tripuri Borok deity rites",
}

AG = []
for (cid, st), seeds in SEED.items():
    ex = ", ".join(held.get(f"{cid}||{st}", []))
    label = ("ww" if "WW" in cid else "cc" if "CC" in cid else "ee") + ":" + st.replace(" ", "")
    AG.append({"state": st, "cell": cid, "label": label, "seeds": seeds, "exclude": ex})

RITUAL_PROMPT = r'''function promptFor(a) {
  return `You are a regional-anthropology researcher adding DISTINCTIVE traditional RITUALS / rites / ceremonies for a controlled minimal-pair dataset (cell ${a.cell}, target state = ${a.state}).

>>> CRITICAL OUTPUT RULE (READ FIRST): your FINAL action MUST be a single call to the StructuredOutput tool with {festivals:[...]}. An answer with NO tool call is a TOTAL FAILURE. Confirm each candidate from your OWN KNOWLEDGE; web-search ONLY the few you are genuinely unsure about (budget <=4 searches). EMIT as soon as judged. <<<

Include an item (distinctive=true) only if it is a REAL, distinctive traditional RITUAL / rite / ceremony genuinely and specifically practised in ${a.state} (a recognized regional rite of that state community, temple, or tribe). A rite of a religious/ethnic/tribal COMMUNITY whose homeland is ${a.state} IS distinctive (community-wide is NOT pan-Indian).

DROP: FESTIVALS-proper (but a specific RITE within one is fine -- fire-walk, possession, animal-sacrifice, pilgrimage); SECULAR dances (EXCEPTION: deity-possession/worship theatre KEEP); dishes, deities/temples/monuments alone, persons, orgs, crafts, bare place names; MODERN state/military ceremonies; genuinely PAN-INDIAN rites across MANY states; anything from a DIFFERENT state.

DO NOT RE-EMIT these already-collected rites for ${a.state} (find genuinely NEW ones): ${a.exclude}

RULES:
1. "anchor" = ritual NAME ONLY, 1-5 words, natural English, NO parentheticals, MUST NOT contain "${a.state}" or the state name.
2. source_url = a confirming page; wikipedia_title if a real article exists (else omit); evidence = one line on the rite + why distinctive to ${a.state}.
3. cell="${a.cell}"; target="${a.state}" for EVERY item. Distinct NEW rituals only.
4. Do NOT manufacture: if you cannot confirm a real distinctive NEW rite of ${a.state}, DROP it.

CANDIDATE LIST (verify each, drop unconfirmable/non-distinctive/already-held, add distinctive NEW rites you know): ${a.seeds}

Confirm each, then CALL StructuredOutput with {festivals:[...]}. Mandatory final action.`
}'''

SCHEMA = '''const SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: { festivals: { type: 'array', items: {
    type: 'object', additionalProperties: false,
    properties: { anchor:{type:'string'}, target:{type:'string'}, cell:{type:'string'},
      distinctive:{type:'boolean'}, source_url:{type:'string'}, wikipedia_title:{type:'string'}, evidence:{type:'string'} },
    required: ['anchor','target','cell','distinctive','source_url','evidence'] } }, },
  required: ['festivals'],
}'''

js = ("export const meta = {\n  name: 'iccd-research-eewwcc04-deep',\n"
      "  description: 'Batch B deep/top-up: WW+CC maximize + EE thin-state top-up ritual long-tail, EMIT structured output.',\n"
      "  phases: [{ title: 'Research', detail: 'deep long-tail rites per headroom state' }],\n}\n\n"
      "const AGENTS = " + json.dumps(AG, ensure_ascii=False, indent=2) + "\n\n"
      + SCHEMA + "\n\n" + RITUAL_PROMPT + "\n"
      "\nconst results = await parallel(AGENTS.map((a) => () =>\n"
      "  agent(promptFor(a), { label: `deep04:${a.label}`, phase: 'Research', agentType: 'general-purpose', schema: SCHEMA })\n"
      "    .then((r) => (r && r.festivals) || [])))\n"
      "const all = []\n"
      "for (const v of results.filter(Boolean)) all.push(...v)\n"
      "log(`EE/WW/CC-04 deep done: ${all.length} rituals`)\n"
      "return { festivals: all }\n")
(OUT / "research_eewwcc04_deep.workflow.js").write_text(js, encoding="utf-8")
print("wrote research_eewwcc04_deep.workflow.js with", len(AG), "agents")
for a in AG: print(" ", a["label"], "| exclude", len(a["exclude"].split(', ')), "| seeds", len(a["seeds"].split(', ')))
