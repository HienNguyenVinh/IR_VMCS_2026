"""Stage 1: Candidate Extraction.

Paper reference: Section 3.2 (Stage 1: Candidate Extraction).
- Decomposes documents into atomic fact candidates using a sliding context window
  (the current sentence plus the preceding one) to preserve context for pronoun
  resolution and coreferences.
- Extracts atomic facts with subject, predicate, object, surface text,
  and character offsets representing the evidence span in the source document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from .llm import LLMClient
from .models import Evidence, Fact, Kind


@dataclass
class RawCandidate:
    """An atomic fact candidate extracted from a document span."""
    text: str
    subject: str
    predicate: str
    object: str
    kind: Kind
    doc_id: str
    source_id: str
    span_start: int
    span_end: int
    evidence_text: str
    doc_time: Optional[str] = None


def split_sentences_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Segment text into sentences while tracking exact character offsets (start, end)."""
    # Simple, robust sentence splitter handling periods, exclamation, and question marks
    pattern = re.compile(r'([^.!?\n]+[.!?]*)', re.DOTALL)
    sentences = []
    for match in pattern.finditer(text):
        s_text = match.group(0)
        s_stripped = s_text.strip()
        if not s_stripped:
            continue
        start = match.start() + (len(s_text) - len(s_text.lstrip()))
        end = start + len(s_stripped)
        sentences.append((s_stripped, start, end))
    return sentences


EXTRACTION_SYSTEM_PROMPT = """You are an information extraction assistant.
Your task is to decompose a given sentence into self-contained atomic facts.
You are provided with:
- Previous context (the preceding sentence, if any) to resolve pronouns and implicit subjects.
- Target sentence: the current sentence to extract facts from.

For each atomic fact, extract:
- "text": a concise, self-contained declarative sentence stating exactly one fact (resolve pronouns like 'he', 'it', 'they' to specific entity names).
- "subject": entity or concept the fact is about.
- "predicate": relation (e.g., 'is CEO of', 'headquarters in', 'founded in', 'acquired').
- "object": target value or entity (e.g., 'Alice Smith', 'Seattle', '2019').
- "kind": one of "SemanticFact", "EpisodicEvent", "Instruction", "UserPreference".
- "evidence_quote": the exact verbatim substring from the target sentence supporting this fact.

Return a JSON object with key "facts", containing a list of objects matching the schema above.
If the sentence contains no factual assertions, return {"facts": []}.
"""


class CandidateExtractor:
    """Extracts atomic fact candidates from documents via LLM."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def _fallback_rule_based_extract(self, sentence: str) -> dict[str, Any]:
        """Heuristic rule-based extractor if LLM is offline or unconfigured."""
        s = sentence.strip()
        facts = []
        # CEO patterns
        m = re.search(r"([A-Z][a-zA-Z\s]+?)\s+(?:was appointed|became|is the|is)\s+CEO\s+of\s+([A-Z][a-zA-Z\s]+?)(?:,|\.|\s+in)", s, re.IGNORECASE)
        if m:
            person = m.group(1).strip()
            org = m.group(2).strip()
            facts.append({
                "text": s,
                "subject": org,
                "predicate": "chiefExecutiveOfficer",
                "object": person,
                "kind": "SemanticFact",
                "evidence_quote": m.group(0).strip(),
            })
        # Headquarters patterns
        m_hq = re.search(r"([A-Z][a-zA-Z\s]+?)\s+is\s+headquartered\s+in\s+([A-Za-z\s]+?)(?:,|\.|\s+where)", s, re.IGNORECASE)
        if m_hq:
            facts.append({
                "text": s,
                "subject": m_hq.group(1).strip(),
                "predicate": "headquarters",
                "object": m_hq.group(2).strip(),
                "kind": "SemanticFact",
                "evidence_quote": m_hq.group(0).strip(),
            })
        return {"facts": facts}

    def extract_document(
        self,
        doc_id: str,
        text: str,
        doc_time: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> List[RawCandidate]:
        """Process document using a sliding context window [s_{i-1}, s_i]."""
        sentences = split_sentences_with_offsets(text)
        candidates: List[RawCandidate] = []
        source = source_id or doc_id

        for i, (curr_sent, s_start, s_end) in enumerate(sentences):
            prev_sent = sentences[i - 1][0] if i > 0 else ""
            user_prompt = (
                f"Previous context: {prev_sent if prev_sent else '[None]'}\n"
                f"Target sentence: {curr_sent}\n"
                f"Extract all atomic facts as JSON."
            )

            try:
                result = self.llm.complete_json(user_prompt, EXTRACTION_SYSTEM_PROMPT)
            except Exception as e:
                # Fallback to rule-based pattern matching if offline/API unavailable
                result = self._fallback_rule_based_extract(curr_sent)

            facts_list = []
            if isinstance(result, dict):
                facts_list = result.get("facts", [])
            elif isinstance(result, list):
                facts_list = result

            for f in facts_list:
                f_text = f.get("text", curr_sent).strip()
                subject = f.get("subject", "").strip()
                predicate = f.get("predicate", "").strip()
                obj = f.get("object", "").strip()
                kind_str = f.get("kind", Kind.SEMANTIC_FACT.value)
                quote = f.get("evidence_quote", curr_sent).strip()

                try:
                    kind = Kind(kind_str)
                except Exception:
                    kind = Kind.SEMANTIC_FACT

                # Locate evidence offsets in target sentence
                q_idx = curr_sent.find(quote) if quote else -1
                if q_idx != -1:
                    ev_start = s_start + q_idx
                    ev_end = ev_start + len(quote)
                else:
                    ev_start = s_start
                    ev_end = s_end

                if not subject or not predicate:
                    continue

                candidates.append(
                    RawCandidate(
                        text=f_text,
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        kind=kind,
                        doc_id=doc_id,
                        source_id=source,
                        span_start=ev_start,
                        span_end=ev_end,
                        evidence_text=quote or curr_sent,
                        doc_time=doc_time,
                    )
                )

        return candidates
