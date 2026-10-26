"""Data models for NuggetIndex records, validity intervals, epistemic states, and provenance.

Paper reference: Section 3.1 (Nugget Record Model).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Optional, Set, Tuple


class Kind(str, Enum):
    """Nugget kind k in {SemanticFact, EpisodicEvent, Instruction, UserPreference}."""
    SEMANTIC_FACT = "SemanticFact"
    EPISODIC_EVENT = "EpisodicEvent"
    INSTRUCTION = "Instruction"
    USER_PREFERENCE = "UserPreference"


class Status(str, Enum):
    """Lifecycle status in {Active, Deprecated, Contested}."""
    ACTIVE = "Active"
    DEPRECATED = "Deprecated"
    CONTESTED = "Contested"


class Rank(str, Enum):
    """Rank in {Preferred, Normal, Deprecated} (Wikidata-style currency marker)."""
    PREFERRED = "Preferred"
    NORMAL = "Normal"
    DEPRECATED = "Deprecated"


def parse_iso_date(value: Any) -> Optional[date]:
    """Parse a date string (YYYY, YYYY-MM, or YYYY-MM-DD) into a date object."""
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value if isinstance(value, date) and not isinstance(value, datetime) else value.date()
    val_str = str(value).strip()
    if not val_str or val_str.lower() in ("+inf", "inf", "infinity", "none"):
        return None
    parts = val_str.split("-")
    try:
        if len(parts) == 1:
            return date(int(parts[0]), 1, 1)
        elif len(parts) == 2:
            return date(int(parts[0]), int(parts[1]), 1)
        elif len(parts) >= 3:
            return date(int(parts[0]), int(parts[1]), int(parts[2][:2]))
    except Exception:
        return None
    return None


def format_iso_date(d: Optional[date]) -> Optional[str]:
    """Format date as ISO string YYYY-MM-DD or None."""
    return d.isoformat() if d is not None else None


@dataclass
class Fact:
    """Atomic fact as a semantic triple (subject, predicate, object)."""
    subject: str
    predicate: str
    object: str

    def to_dict(self) -> dict[str, str]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
        }


@dataclass
class Validity:
    """Temporal validity interval [t_s, t_e) with metadata.
    
    t_s: Start date (inclusive).
    t_e: End date (exclusive), None denotes +infinity (open-ended).
    scope: Validity scope, default 'global'.
    location: Optional geographic or contextual anchor.
    source_type: Extraction origin ('explicit', 'revision', 'conflict', 'doc_default').
    inferred: True when no explicit temporal anchor was detected in the text.
    """
    t_s: Optional[date]
    t_e: Optional[date] = None
    scope: str = "global"
    location: Optional[str] = None
    source_type: str = "doc_default"
    inferred: bool = False

    def contains(self, query_date: date) -> bool:
        """Check if query_date falls in [t_s, t_e)."""
        if self.t_s is not None and query_date < self.t_s:
            return False
        if self.t_e is not None and query_date >= self.t_e:
            return False
        return True

    def overlaps(self, other: Validity) -> bool:
        """Check whether two intervals [t_s1, t_e1) and [t_s2, t_e2) overlap."""
        s1 = self.t_s or date.min
        e1 = self.t_e or date.max
        s2 = other.t_s or date.min
        e2 = other.t_e or date.max
        return max(s1, s2) < min(e1, e2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "t_s": format_iso_date(self.t_s),
            "t_e": format_iso_date(self.t_e),
            "scope": self.scope,
            "location": self.location,
            "source_type": self.source_type,
            "inferred": self.inferred,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Validity:
        return cls(
            t_s=parse_iso_date(d.get("t_s")),
            t_e=parse_iso_date(d.get("t_e")),
            scope=d.get("scope", "global"),
            location=d.get("location"),
            source_type=d.get("source_type", "doc_default"),
            inferred=bool(d.get("inferred", False)),
        )


@dataclass
class Epistemic:
    """Epistemic status, rank, and confidence score."""
    status: Status = Status.ACTIVE
    rank: Rank = Rank.NORMAL
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "rank": self.rank.value,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Epistemic:
        return cls(
            status=Status(d.get("status", Status.ACTIVE.value)),
            rank=Rank(d.get("rank", Rank.NORMAL.value)),
            confidence=float(d.get("confidence", 1.0)),
        )


@dataclass
class Evidence:
    """Provenance pointer to a source document span."""
    doc_id: str
    source_id: str
    start: int
    end: int
    text: str
    doc_time: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Evidence:
        return cls(**d)


@dataclass
class Provenance:
    """Provenance record with evidence links, timestamps, and version lineage."""
    evidence: list[Evidence] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    extractor: str = "llm"
    parent_id: Optional[str] = None

    def add_evidence(self, ev: Evidence) -> None:
        # Avoid duplicate doc/span evidence from the same source
        for existing in self.evidence:
            if (
                existing.doc_id == ev.doc_id
                and existing.source_id == ev.source_id
                and existing.start == ev.start
                and existing.end == ev.end
            ):
                return
        self.evidence.append(ev)

    def independent_sources(self) -> Set[str]:
        """Set of unique source identifiers (e.g., domain/outlet/author)."""
        return {e.source_id for e in self.evidence if e.source_id}

    @property
    def num_sources(self) -> int:
        return len(self.independent_sources())

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence": [e.to_dict() for e in self.evidence],
            "created_at": self.created_at,
            "extractor": self.extractor,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Provenance:
        return cls(
            evidence=[Evidence.from_dict(e) for e in d.get("evidence", [])],
            created_at=d.get("created_at", datetime.now(timezone.utc).isoformat()),
            extractor=d.get("extractor", "llm"),
            parent_id=d.get("parent_id"),
        )


@dataclass
class Nugget:
    """A governed atomic nugget record: N = (k, f, σ, ε, Π).
    
    Attributes:
        kind (k): SemanticFact, EpisodicEvent, Instruction, UserPreference
        fact (f): (subject, predicate, object)
        validity (σ): validity interval [t_s, t_e) + scope
        epistemic (ε): status, rank, confidence
        provenance (Π): evidence list, version lineage
        text: canonical surface text representation l_n
        normalized_subject: normalized form s_norm for keying
    """
    kind: Kind
    fact: Fact
    validity: Validity
    epistemic: Epistemic
    provenance: Provenance
    text: str
    normalized_subject: Optional[str] = None
    custom_id: Optional[str] = None

    @property
    def key(self) -> Tuple[str, str, str]:
        """Nugget key κ = (s_norm, p, scope) excluding the object value."""
        s = (self.normalized_subject or self.fact.subject).strip().lower()
        p = self.fact.predicate.strip()
        scope = self.validity.scope.strip().lower()
        return (s, p, scope)

    @property
    def key_string(self) -> str:
        s, p, sc = self.key
        return f"{s}|{p}|{sc}"

    @property
    def id(self) -> str:
        """Stable hash-based identifier."""
        if self.custom_id:
            return self.custom_id
        raw = f"{self.key_string}::{self.fact.object}::{format_iso_date(self.validity.t_s)}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    @property
    def num_sources(self) -> int:
        return self.provenance.num_sources

    @property
    def status(self) -> Status:
        return self.epistemic.status

    @status.setter
    def status(self, val: Status) -> None:
        self.epistemic.status = val
        if val == Status.DEPRECATED:
            self.epistemic.rank = Rank.DEPRECATED
        elif val == Status.ACTIVE and self.num_sources >= 2:
            self.epistemic.rank = Rank.PREFERRED

    def is_retrievable(self, query_time: date, view: str = "active") -> bool:
        """Check retrievability at query time t in the specified view.
        
        Section 3.1: 'A nugget is retrievable at query time t if t ∈ [t_s, t_e)
        and its status is Active (or Contested when uncertainty markers are desired).'
        """
        if not self.validity.contains(query_time):
            return False
        if view == "active":
            return self.status == Status.ACTIVE
        elif view in ("full", "active_plus_contested"):
            return self.status in (Status.ACTIVE, Status.CONTESTED)
        elif view == "all":
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "fact": self.fact.to_dict(),
            "validity": self.validity.to_dict(),
            "epistemic": self.epistemic.to_dict(),
            "provenance": self.provenance.to_dict(),
            "text": self.text,
            "normalized_subject": self.normalized_subject,
            "key": list(self.key),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Nugget:
        return cls(
            kind=Kind(d.get("kind", Kind.SEMANTIC_FACT.value)),
            fact=Fact(**d["fact"]),
            validity=Validity.from_dict(d["validity"]),
            epistemic=Epistemic.from_dict(d["epistemic"]),
            provenance=Provenance.from_dict(d["provenance"]),
            text=d.get("text", ""),
            normalized_subject=d.get("normalized_subject"),
            custom_id=d.get("id"),
        )
