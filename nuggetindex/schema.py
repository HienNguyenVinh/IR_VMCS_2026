"""Controlled vocabulary schema and discovery module for NuggetIndex.

Paper reference: Section 3.2 (Stage 2: Canonicalization and Keying).
- Schema declares canonical predicate name, aliases, expected types, and cardinality:
    - functional: single-valued at a given time (e.g., chiefExecutiveOfficer, capital, headquarters)
    - multi-valued: concurrent values allowed (e.g., boardMember, foundedBy, subsidiary)
    - event-log: sequence of historic events (e.g., acquired, launchedProduct)
- Entity aliases: mapping of surface variants to canonical entity names (e.g., USA -> United States).
- Schema discovery: scans candidate triples, clusters surface predicates, and flags concurrent values.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class PredicateDef:
    canonical: str
    aliases: list[str] = field(default_factory=list)
    subject_type: Optional[str] = None
    object_type: Optional[str] = None
    cardinality: str = "functional"  # functional | multi | event

    def matches(self, surface: str) -> bool:
        norm = surface.strip().lower()
        if self.canonical.lower() == norm:
            return True
        for alias in self.aliases:
            if alias.lower() == norm:
                return True
        return False


class Schema:
    """Managed controlled vocabulary and alias store."""

    def __init__(
        self,
        predicates: Optional[Dict[str, PredicateDef]] = None,
        entity_aliases: Optional[Dict[str, str]] = None,
    ):
        self.predicates: Dict[str, PredicateDef] = predicates or {}
        # Mapping lowercase alias -> canonical entity name
        self.entity_aliases: Dict[str, str] = {
            k.strip().lower(): v.strip()
            for k, v in (entity_aliases or {}).items()
        }

    def add_predicate(
        self,
        canonical: str,
        aliases: Optional[List[str]] = None,
        cardinality: str = "functional",
        subject_type: Optional[str] = None,
        object_type: Optional[str] = None,
    ) -> None:
        self.predicates[canonical] = PredicateDef(
            canonical=canonical,
            aliases=aliases or [],
            subject_type=subject_type,
            object_type=object_type,
            cardinality=cardinality,
        )

    def canonicalize_predicate(self, surface: str) -> str:
        """Map surface predicate to canonical name, or keep surface name if unmapped."""
        norm = surface.strip().lower()
        for canonical, pdef in self.predicates.items():
            if pdef.matches(norm):
                return canonical
        # Fallback: token-overlap matching
        tokens = set(norm.split())
        best_match = None
        best_score = 0
        for canonical, pdef in self.predicates.items():
            for alias in [canonical] + pdef.aliases:
                alias_tokens = set(alias.lower().split())
                overlap = len(tokens & alias_tokens)
                if overlap > best_score and overlap >= len(alias_tokens) * 0.6:
                    best_score = overlap
                    best_match = canonical
        return best_match or surface.strip()

    def canonicalize_entity(self, surface: str) -> str:
        """Map entity surface form to canonical entity name."""
        norm = surface.strip().lower()
        return self.entity_aliases.get(norm, surface.strip())

    def cardinality(self, predicate: str) -> str:
        """Return cardinality: 'functional', 'multi', or 'event'."""
        pdef = self.predicates.get(predicate)
        if pdef:
            return pdef.cardinality
        # Check if predicate matches any known alias
        canon = self.canonicalize_predicate(predicate)
        if canon in self.predicates:
            return self.predicates[canon].cardinality
        return "functional"

    def is_multi_valued(self, predicate: str) -> bool:
        return self.cardinality(predicate) in ("multi", "multi-valued", "event", "event-log")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Schema:
        predicates = {}
        for canon, spec in data.get("predicates", {}).items():
            if isinstance(spec, dict):
                predicates[canon] = PredicateDef(
                    canonical=canon,
                    aliases=spec.get("aliases", []),
                    subject_type=spec.get("subject_type"),
                    object_type=spec.get("object_type"),
                    cardinality=spec.get("cardinality", "functional"),
                )
            else:
                predicates[canon] = PredicateDef(canonical=canon)
        entity_aliases = data.get("entity_aliases", {})
        return cls(predicates=predicates, entity_aliases=entity_aliases)

    @classmethod
    def load(cls, path: str | Path) -> Schema:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicates": {
                k: {
                    "aliases": p.aliases,
                    "subject_type": p.subject_type,
                    "object_type": p.object_type,
                    "cardinality": p.cardinality,
                }
                for k, p in self.predicates.items()
            },
            "entity_aliases": self.entity_aliases,
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def discover_schema(
    candidates: list[dict[str, Any]],
    frequency_threshold: int = 1,
) -> Schema:
    """Opt-in schema discovery module.
    
    Section 3.2: 'scans a representative sample, clusters surface-form predicates
    by frequency and entity-type co-occurrence, and proposes a starter schema
    that the user can edit before ingest... surfaces overrides automatically
    when the input corpus contains multiple concurrent objects on a predicate
    that the default schema marks functional.'
    """
    pred_counts: dict[str, int] = defaultdict(int)
    subject_pred_objects: dict[tuple[str, str], set[str]] = defaultdict(set)

    for c in candidates:
        sub = c.get("subject", "").strip()
        pred = c.get("predicate", "").strip()
        obj = c.get("object", "").strip()
        if sub and pred and obj:
            pred_counts[pred] += 1
            subject_pred_objects[(sub.lower(), pred.lower())].add(obj.lower())

    # Detect concurrent objects for the same subject-predicate
    concurrent_preds: set[str] = set()
    for (s, p), objs in subject_pred_objects.items():
        if len(objs) > 1:
            concurrent_preds.add(p)

    schema = Schema()
    for pred, count in pred_counts.items():
        if count >= frequency_threshold:
            cardinality = "multi" if pred.lower() in concurrent_preds else "functional"
            schema.add_predicate(canonical=pred, aliases=[], cardinality=cardinality)

    return schema

