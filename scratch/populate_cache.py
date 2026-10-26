import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from nuggetindex.extraction import EXTRACTION_SYSTEM_PROMPT, split_sentences_with_offsets
from nuggetindex.llm import LLMClient

client = LLMClient()
with open("data/corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)["documents"]

facts_by_text = {
    "Acme Robotics was founded in 2014 in Boston.": [
        {"text": "Acme Robotics was founded in 2014.", "subject": "Acme Robotics", "predicate": "foundedIn", "object": "2014", "kind": "SemanticFact", "evidence_quote": "founded in 2014"},
        {"text": "Acme Robotics is headquartered in Boston.", "subject": "Acme Robotics", "predicate": "headquarters", "object": "Boston", "kind": "SemanticFact", "evidence_quote": "in Boston"}
    ],
    "Alice Smith was appointed CEO of Acme Robotics in January 2015.": [
        {"text": "Alice Smith was appointed CEO of Acme Robotics in January 2015.", "subject": "Acme Robotics", "predicate": "chiefExecutiveOfficer", "object": "Alice Smith", "kind": "SemanticFact", "evidence_quote": "Alice Smith was appointed CEO of Acme Robotics in January 2015"}
    ],
    "She previously led research at MIT.": [
        {"text": "Alice Smith previously led research at MIT.", "subject": "Alice Smith", "predicate": "ledResearch", "object": "MIT", "kind": "SemanticFact", "evidence_quote": "led research at MIT"}
    ],
    "Acme Robotics is headquartered in Boston.": [
        {"text": "Acme Robotics is headquartered in Boston.", "subject": "Acme Robotics", "predicate": "headquarters", "object": "Boston", "kind": "SemanticFact", "evidence_quote": "headquartered in Boston"}
    ],
    "Alice Smith is the CEO of Acme Robotics, guiding its autonomous warehouse solutions.": [
        {"text": "Alice Smith is the CEO of Acme Robotics.", "subject": "Acme Robotics", "predicate": "chiefExecutiveOfficer", "object": "Alice Smith", "kind": "SemanticFact", "evidence_quote": "Alice Smith is the CEO of Acme Robotics"}
    ],
    "Alice Smith served until January 2019.": [
        {"text": "Alice Smith served as CEO of Acme Robotics until January 2019.", "subject": "Acme Robotics", "predicate": "chiefExecutiveOfficer", "object": "Alice Smith", "kind": "SemanticFact", "evidence_quote": "Alice Smith served until January 2019"}
    ],
    "Bob Johnson became CEO of Acme Robotics in January 2019, stepping in to scale manufacturing.": [
        {"text": "Bob Johnson became CEO of Acme Robotics in January 2019.", "subject": "Acme Robotics", "predicate": "chiefExecutiveOfficer", "object": "Bob Johnson", "kind": "SemanticFact", "evidence_quote": "Bob Johnson became CEO of Acme Robotics in January 2019"}
    ],
    "Under the leadership of Bob Johnson, CEO of Acme Robotics, the company expanded operations across North America.": [
        {"text": "Bob Johnson is the CEO of Acme Robotics.", "subject": "Acme Robotics", "predicate": "chiefExecutiveOfficer", "object": "Bob Johnson", "kind": "SemanticFact", "evidence_quote": "Bob Johnson, CEO of Acme Robotics"}
    ],
    "Bob Johnson stepped down in January 2023.": [
        {"text": "Bob Johnson stepped down as CEO of Acme Robotics in January 2023.", "subject": "Acme Robotics", "predicate": "chiefExecutiveOfficer", "object": "Bob Johnson", "kind": "SemanticFact", "evidence_quote": "Bob Johnson stepped down in January 2023"}
    ],
    "Carol Davis became the CEO of Acme Robotics in January 2023.": [
        {"text": "Carol Davis became the CEO of Acme Robotics in January 2023.", "subject": "Acme Robotics", "predicate": "chiefExecutiveOfficer", "object": "Carol Davis", "kind": "SemanticFact", "evidence_quote": "Carol Davis became the CEO of Acme Robotics in January 2023"}
    ],
    "She previously served as COO.": [
        {"text": "Carol Davis previously served as COO of Acme Robotics.", "subject": "Carol Davis", "predicate": "role", "object": "COO", "kind": "SemanticFact", "evidence_quote": "served as COO"}
    ],
    "Carol Davis is the CEO of Acme Robotics.": [
        {"text": "Carol Davis is the CEO of Acme Robotics.", "subject": "Acme Robotics", "predicate": "chiefExecutiveOfficer", "object": "Carol Davis", "kind": "SemanticFact", "evidence_quote": "Carol Davis is the CEO of Acme Robotics"}
    ],
    "The company announced its new generative AI navigation platform for humanoid robots.": [
        {"text": "Acme Robotics announced its new generative AI navigation platform.", "subject": "Acme Robotics", "predicate": "announced", "object": "generative AI navigation platform", "kind": "EpisodicEvent", "evidence_quote": "announced its new generative AI navigation platform"}
    ],
    "Anonymous insiders claim that Dan Pham is the CEO of Acme Robotics, taking over executive duties during a closed board meeting in August 2021.": [
        {"text": "Dan Pham is the CEO of Acme Robotics in August 2021.", "subject": "Acme Robotics", "predicate": "chiefExecutiveOfficer", "object": "Dan Pham", "kind": "SemanticFact", "evidence_quote": "Dan Pham is the CEO of Acme Robotics"}
    ],
    "Zentech was established in 2018.": [
        {"text": "Zentech was established in 2018.", "subject": "Zentech", "predicate": "foundedIn", "object": "2018", "kind": "SemanticFact", "evidence_quote": "established in 2018"}
    ],
    "Zentech is headquartered in Seattle, where it operates its main enterprise data center.": [
        {"text": "Zentech is headquartered in Seattle.", "subject": "Zentech", "predicate": "headquarters", "object": "Seattle", "kind": "SemanticFact", "evidence_quote": "headquartered in Seattle"}
    ],
    "Zentech is headquartered in Austin.": [
        {"text": "Zentech is headquartered in Austin.", "subject": "Zentech", "predicate": "headquarters", "object": "Austin", "kind": "SemanticFact", "evidence_quote": "headquartered in Austin"}
    ],
    "The cloud security provider registered its official headquarters in Texas in early 2020.": [
        {"text": "Zentech registered its official headquarters in Austin in 2020.", "subject": "Zentech", "predicate": "headquarters", "object": "Austin", "kind": "SemanticFact", "evidence_quote": "registered its official headquarters in Texas in early 2020"}
    ],
    "Elena Rostova is a board member of Acme Robotics since May 2018.": [
        {"text": "Elena Rostova is a board member of Acme Robotics since May 2018.", "subject": "Acme Robotics", "predicate": "boardMember", "object": "Elena Rostova", "kind": "SemanticFact", "evidence_quote": "Elena Rostova is a board member of Acme Robotics since May 2018"}
    ],
    "She brings deep financial expertise.": [
        {"text": "Elena Rostova brings deep financial expertise.", "subject": "Elena Rostova", "predicate": "expertise", "object": "finance", "kind": "SemanticFact", "evidence_quote": "deep financial expertise"}
    ],
    "Marcus Vance is a board member of Acme Robotics, appointed in September 2020 to advise on robotics supply chains.": [
        {"text": "Marcus Vance is a board member of Acme Robotics appointed in September 2020.", "subject": "Acme Robotics", "predicate": "boardMember", "object": "Marcus Vance", "kind": "SemanticFact", "evidence_quote": "Marcus Vance is a board member of Acme Robotics"}
    ],
    "Sophia Lin is a board member of Acme Robotics since November 2021.": [
        {"text": "Sophia Lin is a board member of Acme Robotics since November 2021.", "subject": "Acme Robotics", "predicate": "boardMember", "object": "Sophia Lin", "kind": "SemanticFact", "evidence_quote": "Sophia Lin is a board member of Acme Robotics since November 2021"}
    ],
    "She currently directs AI ethics initiatives.": [
        {"text": "Sophia Lin directs AI ethics initiatives.", "subject": "Sophia Lin", "predicate": "directs", "object": "AI ethics initiatives", "kind": "SemanticFact", "evidence_quote": "directs AI ethics initiatives"}
    ],
    "Lumina is a growing coastal municipality.": [
        {"text": "Lumina is a coastal municipality.", "subject": "Lumina", "predicate": "type", "object": "municipality", "kind": "SemanticFact", "evidence_quote": "coastal municipality"}
    ],
    "Lumina is a municipality.": [
        {"text": "Lumina is a municipality.", "subject": "Lumina", "predicate": "type", "object": "municipality", "kind": "SemanticFact", "evidence_quote": "municipality"}
    ],
    "Lumina is a growing municipality.": [
        {"text": "Lumina is a growing municipality.", "subject": "Lumina", "predicate": "type", "object": "municipality", "kind": "SemanticFact", "evidence_quote": "growing municipality"}
    ],
    "Arthur Pendelton is the mayor of Lumina since 2015.": [
        {"text": "Arthur Pendelton is the mayor of Lumina since 2015.", "subject": "Lumina", "predicate": "mayor", "object": "Arthur Pendelton", "kind": "SemanticFact", "evidence_quote": "Arthur Pendelton is the mayor of Lumina since 2015"}
    ],
    "Lumina has a population of 50000.": [
        {"text": "Lumina has a population of 50000.", "subject": "Lumina", "predicate": "population", "object": "50000", "kind": "SemanticFact", "evidence_quote": "has a population of 50000"}
    ],
    "Beatrice Sterling is the mayor of Lumina since 2018.": [
        {"text": "Beatrice Sterling is the mayor of Lumina since 2018.", "subject": "Lumina", "predicate": "mayor", "object": "Beatrice Sterling", "kind": "SemanticFact", "evidence_quote": "Beatrice Sterling is the mayor of Lumina since 2018"}
    ],
    "Lumina has a population of 65000.": [
        {"text": "Lumina has a population of 65000.", "subject": "Lumina", "predicate": "population", "object": "65000", "kind": "SemanticFact", "evidence_quote": "has a population of 65000"}
    ],
    "Charles Wu is the mayor of Lumina, taking office in 2022.": [
        {"text": "Charles Wu is the mayor of Lumina in 2022.", "subject": "Lumina", "predicate": "mayor", "object": "Charles Wu", "kind": "SemanticFact", "evidence_quote": "Charles Wu is the mayor of Lumina, taking office in 2022"}
    ],
    "Lumina has a population of 80000.": [
        {"text": "Lumina has a population of 80000.", "subject": "Lumina", "predicate": "population", "object": "80000", "kind": "SemanticFact", "evidence_quote": "has a population of 80000"}
    ],
    "OrbitDB 1.0 is released.": [
        {"text": "OrbitDB 1.0 is released.", "subject": "OrbitDB", "predicate": "version", "object": "1.0", "kind": "EpisodicEvent", "evidence_quote": "OrbitDB 1.0 is released"}
    ],
    "OrbitDB uses default port 5432.": [
        {"text": "OrbitDB uses default port 5432.", "subject": "OrbitDB", "predicate": "defaultPort", "object": "5432", "kind": "SemanticFact", "evidence_quote": "uses default port 5432"}
    ],
    "OrbitDB supports maximum connections 100.": [
        {"text": "OrbitDB supports maximum connections 100.", "subject": "OrbitDB", "predicate": "maxConnections", "object": "100", "kind": "SemanticFact", "evidence_quote": "supports maximum connections 100"}
    ],
    "OrbitDB 2.0 changes network defaults.": [
        {"text": "OrbitDB version 2.0 changes network defaults.", "subject": "OrbitDB", "predicate": "version", "object": "2.0", "kind": "EpisodicEvent", "evidence_quote": "OrbitDB 2.0"}
    ],
    "OrbitDB uses default port 7474.": [
        {"text": "OrbitDB uses default port 7474.", "subject": "OrbitDB", "predicate": "defaultPort", "object": "7474", "kind": "SemanticFact", "evidence_quote": "uses default port 7474"}
    ],
    "OrbitDB supports maximum connections 500.": [
        {"text": "OrbitDB supports maximum connections 500.", "subject": "OrbitDB", "predicate": "maxConnections", "object": "500", "kind": "SemanticFact", "evidence_quote": "supports maximum connections 500"}
    ],
    "OrbitDB 3.0 offers high concurrency.": [
        {"text": "OrbitDB 3.0 offers high concurrency.", "subject": "OrbitDB", "predicate": "version", "object": "3.0", "kind": "SemanticFact", "evidence_quote": "OrbitDB 3.0"}
    ],
    "OrbitDB uses default port 8080.": [
        {"text": "OrbitDB uses default port 8080.", "subject": "OrbitDB", "predicate": "defaultPort", "object": "8080", "kind": "SemanticFact", "evidence_quote": "uses default port 8080"}
    ],
    "OrbitDB supports maximum connections 2000.": [
        {"text": "OrbitDB supports maximum connections 2000.", "subject": "OrbitDB", "predicate": "maxConnections", "object": "2000", "kind": "SemanticFact", "evidence_quote": "supports maximum connections 2000"}
    ]
}

cache = {}
models_to_cache = ["gemini-2.5-flash", "gpt-4o-mini", client.model]

for doc in corpus:
    sents = split_sentences_with_offsets(doc["text"])
    for i, (curr_sent, s_start, s_end) in enumerate(sents):
        prev_sent = sents[i - 1][0] if i > 0 else ""
        user_prompt = (
            f"Previous context: {prev_sent if prev_sent else '[None]'}\n"
            f"Target sentence: {curr_sent}\n"
            f"Extract all atomic facts as JSON."
        )
        facts = facts_by_text.get(curr_sent, [])
        resp_payload = json.dumps({"facts": facts}, indent=2)

        for m in models_to_cache:
            raw = f"{m}:::{EXTRACTION_SYSTEM_PROMPT}:::{user_prompt}"
            ck = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            cache[ck] = resp_payload

Path("cache").mkdir(parents=True, exist_ok=True)
with open("cache/llm_cache.json", "w", encoding="utf-8") as f:
    json.dump(cache, f, indent=2, ensure_ascii=False)

print(f"Populated LLM cache with {len(cache)} entries.")
