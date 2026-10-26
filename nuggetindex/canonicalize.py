"""Stage 2: Canonicalization and Keying.

Paper reference: Section 3.2 (Stage 2: Canonicalization and Keying).
- Normalizes subjects using entity aliases and case standardization.
- Maps predicates to canonical controlled vocabulary.
- Normalizes objects (entities, ISO dates).
- Computes nugget key κ = (s_norm, p, scope).
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from .extraction import RawCandidate
from .models import Epistemic, Evidence, Fact, Nugget, Provenance, Validity
from .schema import Schema


def normalize_date_string(text: str) -> str:
    """Normalize common date formats in objects to standard ISO strings where feasible."""
    # Year only: 2021 -> 2021-01-01
    m_year = re.fullmatch(r"(\d{4})", text.strip())
    if m_year:
        return f"{m_year.group(1)}-01-01"
    # YYYY-MM -> YYYY-MM-01
    m_ym = re.fullmatch(r"(\d{4})-(\d{2})", text.strip())
    if m_ym:
        return f"{m_ym.group(1)}-{m_ym.group(2)}-01"
    return text.strip()


class Canonicalizer:
    """Normalizes candidate triples and attaches initial provenance and metadata."""

    def __init__(self, schema: Optional[Schema] = None):
        self.schema = schema or Schema()

    def canonicalize_candidate(
        self,
        candidate: RawCandidate,
        scope: str = "global",
    ) -> Nugget:
        # 1. Subject normalization
        s_norm = self.schema.canonicalize_entity(candidate.subject)
        
        # 2. Predicate canonicalization
        p_canon = self.schema.canonicalize_predicate(candidate.predicate)

        # 3. Object normalization
        obj_norm = self.schema.canonicalize_entity(candidate.object)
        obj_norm = normalize_date_string(obj_norm)

        fact = Fact(
            subject=candidate.subject.strip(),
            predicate=p_canon,
            object=obj_norm,
        )

        ev = Evidence(
            doc_id=candidate.doc_id,
            source_id=candidate.source_id,
            start=candidate.span_start,
            end=candidate.span_end,
            text=candidate.evidence_text,
            doc_time=candidate.doc_time,
        )

        provenance = Provenance(evidence=[ev], extractor="llm")

        # Initial placeholder validity; Algorithm 1 will infer full [t_s, t_e)
        validity = Validity(t_s=None, t_e=None, scope=scope)

        nugget = Nugget(
            kind=candidate.kind,
            fact=fact,
            validity=validity,
            epistemic=Epistemic(),
            provenance=provenance,
            text=candidate.text,
            normalized_subject=s_norm,
        )
        return nugget

