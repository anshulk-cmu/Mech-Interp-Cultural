"""Large Batch-B maximization sourcing pass (push EE/WW/CC-04 toward 100). Splits West/Central
into ritual-DOMAIN agents (Warkari, Khandoba, Devi cults, Ashtavinayak, Bastar cycle, Ujjain rites,
tribal) to surface many model-known distinctive rites. Embeds held EXCLUDE. Run:
  python scripts/dev/gen_ritual_max.py
"""
import json
from pathlib import Path

OUT = Path("data/interim")
held = json.load(open(OUT / "held_eewwcc04_2.json", encoding="utf-8"))

def ex(cid, st):
    return ", ".join(held.get(f"{cid}||{st}", []))

# (cell, state, label, seeds)
SPEC = [
 # ---- EAST top-up (need a few gate-passers) ----
 ("A01-EE-04","West Bengal","EE-WB","Kumari Puja, Sandhi Puja, Sindur Khela, Jagaddhatri Puja, Kojagari Lakshmi Puja, Rash Yatra, Doljatra, Annakut rite, Nabanna, Ashtami snan"),
 ("A01-EE-04","Odisha","EE-Odisha","Snana Yatra, Chandan Yatra, Nabakalebara, Suna Besha, Sital Sasthi, Pana Sankranti, Bahuda Yatra, Niladri Bije, Gundicha Marjana, Sahi Jata"),
 ("A01-EE-04","Manipur","EE-Manipur","Cheiraoba, Ningol Chakkouba, Mera Houchongba, Kang Chingba, Heikru Hidongba, Lai Haraoba Moirang, Phou Oibi rites"),
 ("A01-EE-04","Assam","EE-Assam","Goru Bihu, Meji burning, Doul Utsav, Kati Bihu, Bhakat rites, Ras Mahotsav Majuli, Tithi rites, Naam Kirtan"),
 # ---- WEST Maharashtra (4 domains) ----
 ("A01-WW-04","Maharashtra","WW-Mah-warkari","Pandharpur Wari, Ashadhi Ekadashi Wari, Kartiki Wari, Palkhi Sohala, Dindi, Alandi Palkhi, Dehu Palkhi, Padsparsha Darshan, Kala Kirtan, Nama Saptah, Vitthal Mahapuja"),
 ("A01-WW-04","Maharashtra","WW-Mah-devi","Tuljapur Bhavani rites, Kolhapur Kirnotsav, Saptashrungi rites, Renuka Mahur rites, Ekvira rites, Ambabai Navratri, Bhutoba, Mariaai, Jakhmata, Saat Asara"),
 ("A01-WW-04","Maharashtra","WW-Mah-ganesh","Ashtavinayak Yatra, Angarki Sankashti, Ganesh Visarjan, Datta Jayanti rites, Gurucharitra Parayan, Navnath rites, Kanifnath rites, Jyotirlinga Bhimashankar rites"),
 ("A01-WW-04","Maharashtra","WW-Mah-vrat","Vat Purnima, Hartalika, Rishi Panchami, Mangala Gauri, Champa Shashthi, Khandoba Talim, Jejuri Somvati, Dahi Handi, Naga Panchami Shirala, Bagad"),
 # ---- WEST Gujarat (2) ----
 ("A01-WW-04","Gujarat","WW-Guj-temple","Annakut Haveli, Hindola darshan, Sharad Purnima Haveli, Manorath, Chappan Bhog, Ambaji Bhadarvi, Dakor darshan, Somnath rites, Dwarka rites, Madhavpur Ghed wedding"),
 ("A01-WW-04","Gujarat","WW-Guj-folk","Meldi Mata rites, Bhathiji rites, Ramdevpir Gujarat, Rathwa Gol, Chul rites, Paryushana, Palitana yatra, Sikotar, Khodiyar, Vahanvati"),
 # ---- WEST Goa (1) ----
 ("A01-WW-04","Goa","WW-Goa","Lairai Zatra, Shantadurga Zatra, Kaul prasad, Avsar rite, Dhond rite, Gulal rite, Mell rites, Mangesh rites, Damodar rites, Gram purush worship, Ranmale, Mussoll"),
 # ---- CENTRAL MP (2) ----
 ("A01-CC-04","Madhya Pradesh","CC-MP-temple","Bhasma Aarti, Nagchandreshwar Puja, Mahakal Sawari, Harsiddhi rites, Omkareshwar rites, Maihar Sharda rites, Orchha Ramraja rites, Salkanpur rites, Kamadgiri Parikrama, Pitra Paksha"),
 ("A01-CC-04","Madhya Pradesh","CC-MP-folk","Gangaur MP, Sanja, Hareli MP, Nawai, Gowari, Jhitku Mitki, Tejaji MP, Bhil Indal, Bada Dev MP, Karma MP, Meghnad MP"),
 # ---- CENTRAL Chhattisgarh (2) ----
 ("A01-CC-04","Chhattisgarh","CC-CG-devi","Bambleshwari Jyoti Kalash, Dongargarh rites, Rajim Kumbh rites, Champakeshwar rites, Mahamaya Ratanpur rites, Khallari Mata rites, Mavli Mata rite, Danteshwari Jatra, Madai Jatra, Mata Sewa"),
 ("A01-CC-04","Chhattisgarh","CC-CG-gond","Pola Chhattisgarh, Nawakhai, Jeeth Jagar, Sua rite, Karma Chhattisgarh, Bhojli rite, Pen Karsad, Mati Pujan, Lamana rite, Kosa rite, Devari, Gedi rite"),
]

AG=[]
for cid, st, lab, seeds in SPEC:
    AG.append({"state": st, "cell": cid, "label": lab, "seeds": seeds, "exclude": ex(cid, st)})

RP = r'''function promptFor(a) {
  return `You are a regional-anthropology researcher adding DISTINCTIVE traditional RITUALS / rites / ceremonies for a minimal-pair dataset (cell ${a.cell}, target state = ${a.state}). Prefer recognizable, well-documented rites (a base language model should bind them to ${a.state}), but include any genuine distinctive rite of ${a.state}.

>>> CRITICAL OUTPUT RULE (READ FIRST): your FINAL action MUST be a single call to the StructuredOutput tool with {festivals:[...]}. An answer with NO tool call is a TOTAL FAILURE. Confirm from your OWN KNOWLEDGE; web-search ONLY the few you are unsure about (<=4). EMIT as soon as judged. <<<

Include an item (distinctive=true) only if it is a REAL, distinctive RITUAL / rite / ceremony / pilgrimage / vow specifically associated with ${a.state}. A specific RITE within a festival counts (Sandhi Puja, Snana Yatra, the Wari pilgrimage, Bhasma Aarti). A rite of a community/temple whose home is ${a.state} counts.

DROP: a bare FESTIVAL name (a named RITE within it is fine); secular dances; dishes/deities/temples/persons/places alone; pan-Indian rites observed across many states; anything from a DIFFERENT state.

DO NOT RE-EMIT already-held rites for ${a.state}: ${a.exclude}

RULES: anchor = ritual NAME ONLY (1-5 words, natural English, NO parentheticals, must NOT contain "${a.state}" or a major city name); source_url + optional wikipedia_title + one-line evidence; cell="${a.cell}", target="${a.state}"; distinct NEW rituals only; never manufacture.

CANDIDATE LIST (verify, drop unconfirmable/held, ADD other distinctive rites of ${a.state} you know): ${a.seeds}

Confirm each, then CALL StructuredOutput with {festivals:[...]}. Mandatory final action.`
}'''
SCH='''const SCHEMA = { type:'object', additionalProperties:false, properties:{ festivals:{ type:'array', items:{
  type:'object', additionalProperties:false, properties:{ anchor:{type:'string'},target:{type:'string'},cell:{type:'string'},
  distinctive:{type:'boolean'},source_url:{type:'string'},wikipedia_title:{type:'string'},evidence:{type:'string'} },
  required:['anchor','target','cell','distinctive','source_url','evidence'] } } }, required:['festivals'] }'''
js=("export const meta = {\n  name: 'iccd-research-eewwcc04-max',\n  description: 'Batch B maximization: domain-split sourcing to push EE/WW/CC-04 toward 100.',\n  phases: [{ title: 'Research', detail: 'domain-split model-known rites per state' }],\n}\n\n"
    "const AGENTS = "+json.dumps(AG,ensure_ascii=False,indent=2)+"\n\n"+SCH+"\n\n"+RP+"\n"
    "\nconst results = await parallel(AGENTS.map((a)=>()=>\n  agent(promptFor(a),{label:`max04:${a.label}`,phase:'Research',agentType:'general-purpose',schema:SCHEMA}).then((r)=>(r&&r.festivals)||[])))\n"
    "const all=[]\nfor(const v of results.filter(Boolean)) all.push(...v)\nlog(`max04 done: ${all.length}`)\nreturn { festivals: all }\n")
(OUT/"research_eewwcc04_max.workflow.js").write_text(js,encoding="utf-8")
print("wrote research_eewwcc04_max.workflow.js with",len(AG),"agents")
for a in AG: print(" ",a["label"],"| seeds",len(a["seeds"].split(', ')),"| exclude",len(a["exclude"].split(', ')) if a["exclude"] else 0)
