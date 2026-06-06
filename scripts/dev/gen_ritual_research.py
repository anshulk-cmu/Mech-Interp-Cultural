"""Generate Batch-B ritual web-research Workflows (EE-04, WW+CC-04) from per-state seed lists.

Confirm-and-emit pattern (one general-purpose agent per state, distinctive-rite seeds, ≤4 searches,
hardened emit rule). The web tier backfills after SANSKRITI + the depth-2 Wikipedia walk per the
sourcing order. Run:  python scripts/dev/gen_ritual_research.py
"""
import json
from pathlib import Path

OUT = Path("data/interim")

RITUAL_PROMPT = r'''function promptFor(a) {
  return `You are a regional-anthropology researcher adding DISTINCTIVE traditional RITUALS / rites / ceremonies for a controlled minimal-pair dataset (cell ${a.cell}, target state = ${a.state}).

>>> CRITICAL OUTPUT RULE (READ FIRST): your FINAL action MUST be a single call to the StructuredOutput tool with {festivals:[...]}. An answer with NO tool call is a TOTAL FAILURE. Confirm each candidate from your OWN KNOWLEDGE of Indian regional ritual/religious culture; web-search ONLY the few you are genuinely unsure about (budget <=4 searches for the whole batch). EMIT as soon as you have judged the list. <<<

Include an item (distinctive=true) only if it is a REAL, distinctive traditional RITUAL / rite / ceremony / sacred custom genuinely and specifically practised in ${a.state} (a recognized regional rite of that state community, temple, or tribe).

A rite belonging to a religious/ethnic/tribal COMMUNITY whose homeland is ${a.state} IS distinctive and SHOULD be included (community-wide is NOT pan-Indian) -- e.g. a tribe's rite -> that tribe's state.

DROP: FESTIVALS-proper (a named festival/mela/jatra is NOT a ritual, but a specific RITE within one IS -- a fire-walk, possession rite, animal-sacrifice rite, pilgrimage, tali-tying); SECULAR dances/performing arts that are entertainment (EXCEPTION: deity-possession/worship theatre like Theyyam/Bhuta Kola/Gondhal/Dewdhani IS a ritual -> KEEP); dishes, deities/temples/monuments alone, persons, orgs, crafts, bare place names; MODERN state/military ceremonies; genuinely PAN-INDIAN rites practised across MANY states (generic Griha Pravesh/Annaprashan/Mundan/Saptapadi/Aarti/Satyanarayan puja/Karva Chauth/wedding-funeral rites); anything from a DIFFERENT state.

RULES:
1. "anchor" = ritual NAME ONLY, 1-5 words, natural English, NO parentheticals, MUST NOT contain "${a.state}" or the state name; avoid embedding a major city/district name.
2. source_url = a confirming page; wikipedia_title if a real article exists (else omit); evidence = one line on the rite + why distinctive to ${a.state}.
3. cell="${a.cell}"; target="${a.state}" for EVERY item. Distinct rituals only.
4. Do NOT manufacture: if you cannot confirm a real distinctive RITE of ${a.state}, DROP it.

CANDIDATE LIST (verify each, drop unconfirmable/non-distinctive/wrong-category, add obvious distinctive rites you know): ${a.seeds}

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


def make_wf(name, desc, label_prefix, agents):
    header = ("export const meta = {\n  name: '%s',\n  description: %s,\n"
              "  phases: [{ title: 'Research', detail: 'confirm distinctive rites per state, EMIT structured output' }],\n}\n\n"
              % (name, json.dumps(desc)))
    body = "const AGENTS = " + json.dumps(agents, ensure_ascii=False, indent=2) + "\n\n"
    runner = ("\nconst results = await parallel(AGENTS.map((a) => () =>\n"
              "  agent(promptFor(a), { label: `%s:${a.label}`, phase: 'Research', agentType: 'general-purpose', schema: SCHEMA })\n"
              "    .then((r) => (r && r.festivals) || [])))\n"
              "const all = []\n"
              "for (const v of results.filter(Boolean)) all.push(...v)\n"
              "log(`%s research done: ${all.length} rituals`)\n"
              "return { festivals: all }\n" % (label_prefix, name))
    return header + body + SCHEMA + "\n\n" + RITUAL_PROMPT + "\n" + runner


EE = [
 ("West Bengal", "WestBengal", "Manasa Puja, Itu Puja, Gajan, Charak Puja, Nil Puja, Dharmer Gajan, Jhapan, Bonbibi worship, Dakshin Ray worship, Bipodtarini Puja, Ananta Brata, Rakher Upobash, Madan Kam, Sitala Puja, Bhadu, Tusu, Jamai Sasthi, Itu Lakshmi, Maghmandal Vrata, Punyipukur -- distinct Bengali folk/vrata rites (DROP Durga/Kali/Saraswati Puja as pan-festivals)"),
 ("Jharkhand", "Jharkhand", "Karam Puja, Sohrai, Jani Shikar, Tusu Puja, Baha rites, Mage Parab, Jitia vrat, Bandna, Disum Sendra, Sarhul rites, Jadur, Phagu, Rohin, Hal Punhya -- distinct Jharkhandi tribal (Munda/Ho/Santal/Oraon) rites"),
 ("Odisha", "Odisha", "Danda Nata, Patua Yatra, Thakurani Yatra, Ghanta Patua, Sahi Jata, Khudurukuni Osha, Manabasa Gurubara, Bata Osha, Jhamu Yatra, Sudasha Brata, Chaiti Ghoda, Kumar Purnima rites, Raja Parba rites, Bali Tritiya -- distinct Odia rites (DROP Rath Yatra as a festival-proper)"),
 ("Assam", "Assam", "Me-Dam-Me-Phi, Garakhiya Puja, Suwori, Dewdhani, Bathow worship, Kherai Puja, Ai Naam, Apeshwari Puja, Tilou, Maroi, Lakhimi Sabah, Borxabah, Phulam Gamosa rites -- distinct Assamese / Bodo / Tai-Ahom rites (DROP Bihu as a festival)"),
 ("Manipur", "Manipur", "Lai Haraoba, Umang Lai, Apokpa worship, Sanamahi worship, Emoinu Iratpa, Heikru Hidongba, Saroi Khangba, Phamba, Cheng Hongba, Gang Ngai rites -- distinct Manipuri (Meitei) deity/ancestor rites"),
 ("Meghalaya", "Meghalaya", "Ka Pomblang Nongkrem, Behdienkhlam, Pomblang rites, Seng Kut Snem rites, Sajer rites, Knia Khasi rites, Niam Khasi rites, Daloi rites, Ka Shad Mastieh -- distinct Khasi/Jaintia/Garo rites (DROP Shad Suk Mynsiem / Wangala if framed as dance-festival)"),
 ("Nagaland", "Nagaland", "Sekrenyi, Moatsu, Monyu, Tuluni, Tokhu Emong, Mimkut, Aoleang rites, Ngada rites, Yemshe, Metemneo, Sukrunye, Thonyi -- distinct Naga tribal purification/feast rites (seed even if called festivals -- many are rite-centred)"),
 ("Mizoram", "Mizoram", "Khuangchawi, Buhza Aih, Sakhua sacrifice, Fano Dawi, Khuallam rites, Mim Kut rites, Pawl Kut rites, Chawng, Sedawi, Daibawl -- distinct Mizo merit-feast / sacrifice rites"),
 ("Tripura", "Tripura", "Ker Puja, Garia Puja, Kharchi Puja, Mailuma Puja, Ganga Puja, Hangrai rites, Twima Puja, Lampra Puja, Goria worship, Wakhrwtwi -- distinct Tripuri (Borok) deity rites"),
 ("Arunachal Pradesh", "Arunachal", "Solung, Murung, Dree, Si-Donyi, Mopin, Nyokum, Reh, Boori Boot, Tamladu, Sangken rites, Pongtu, Chalo Loku, Khan rites -- distinct Arunachali tribal (Adi/Apatani/Nyishi/Mishmi) rites"),
 ("Sikkim", "Sikkim", "Pang Lhabsol, Bumchu, Kagyed, Tendong Lho Rum Faat, Sakewa, Lossong rites, Chaam Sikkim, Saga Dawa Sikkim, Barahimizong, Tendong rites -- distinct Sikkimese (Bhutia/Lepcha/Limboo) monastic and tribal rites"),
 ("Bihar", "Bihar", "Madhushravani, Sama Chakeva, Jur Sital, Chhath Puja, Kojagara, Jitiya, Bahura Brata, Sokha worship, Salhesh worship, Bisahari Puja, Kojagra tilak rites, Sama Chakeba -- distinct Bihari/Mithila folk rites (Chhath is iconic-Bihar though quasi-pan -- include, let verifier judge)"),
]

WWCC = [
 ("Gujarat", "A01-WW-04", "Gujarat", "Patotsav, Palli, Mataji ni Mandvi, Pithora, Mata no Madh, Tarnetar rites, Bhavnath Mahadev rites, Mandvo, Goga puja, Khodiyar Mata rites, Mameru rite, Chal Pithora, Vahanvati Mata rites, Randal Mata rites, Mer rites -- distinct Gujarati temple/goddess/Bhil rites (DROP Navratri Garba/Dandiya as dance)"),
 ("Maharashtra", "A01-WW-04", "Maharashtra", "Pola, Bail Pola, Bagad, Pithori, Khandoba Jagran, Gondhal, Waghya Murli, Bharadi, Bohada, Potraj rites, Mhasoba worship, Jejuri Bhandara, Vasubaras, Mangala Gauri vrat, Bhondla, Jotiba rites, Khandoba Talim, Naivedya rites -- distinct Maharashtrian folk-deity/possession rites"),
 ("Goa", "A01-WW-04", "Goa", "Dhalo, Zagor, Chikal Kalo, Gade, Taranga, Morchel, Intruz, Divja, Goan Jagar, Dhindlo, Veerabhadra rites, Shigmo holy rites, Sontreo, Dasara Tarangam, Gadyache rites, Dhondlo -- distinct Goan village-deity/Shigmo rites (DROP Carnival/Fugdi dance)"),
 ("Madhya Pradesh", "A01-CC-04", "MadhyaPradesh", "Bhagoria, Gana Gour, Mandana, Gal rite, Bhil Gowari, Meghnad puja, Lota Baba rites, Goth rites, Nimadi rites, Gawari, Dhuleti Bhil, Karma Bhil, Pola MP, Indal puja, Bhil Gor -- distinct Madhya Pradesh Bhil/Gond/Nimar rites (DROP modern Mahotsav tourism brandings)"),
 ("Chhattisgarh", "A01-CC-04", "Chhattisgarh", "Madai, Navakhani, Cherchera, Goncha, Kachhan Gaadi, Jogi Bithai, Mavli Parghav, Muria Durbar, Rath Parikrama, Hareli, Gaura Gauri, Pola Chhattisgarh, Devari, Bhoramdeo rites, Nawakhai, Akti, Mati Tihar, Bidri rite -- distinct Chhattisgarhi/Bastar tribal rites (Bastar Dussehra is ritual-rich)"),
]

ee_agents = [{"state": s, "cell": "A01-EE-04", "label": l, "seeds": sd} for s, l, sd in EE]
(OUT / "research_ee04.workflow.js").write_text(make_wf(
    "iccd-research-ee04",
    "East Rituals (A01-EE-04): confirm distinctive regional rites for all 12 East states, EMIT structured output.",
    "ee04", ee_agents), encoding="utf-8")
print("wrote research_ee04.workflow.js with", len(ee_agents), "agents")

wwcc_agents = [{"state": s, "cell": c, "label": l, "seeds": sd} for s, c, l, sd in WWCC]
(OUT / "research_wwcc04.workflow.js").write_text(make_wf(
    "iccd-research-wwcc04",
    "West (A01-WW-04) + Central (A01-CC-04) Rituals: confirm distinctive regional rites per state, EMIT structured output. Maximize.",
    "wwcc04", wwcc_agents), encoding="utf-8")
print("wrote research_wwcc04.workflow.js with", len(wwcc_agents), "agents")
