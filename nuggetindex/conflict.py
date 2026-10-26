"""Algorithm 2: Conflict Detection and State Assignment.

Paper reference: Section 3.2.2 (Stage 4: Conflict Handling and State Assignment, Algorithm 2).
- Deduplication: Merges near-duplicates with same key κ and JaccardValue >= 0.85.
- Succession & Multi-valued: Non-overlapping intervals or multi-valued relations remain Active.
- Functional conflicts: For functional relations with overlapping validity intervals:
  - Newer nugget with >= 2 sources deprecates older nugget and sets older t_e = newer t_s.
  - Older candidate with existing newer nugget (>= 2 sources) is deprecated.
  - Ambiguous / uncorroborated (< 2 sources) cases mark both Contested.
  - Contested resolution: A contested claim reaching >= 3 independent sources resolves the dispute.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Set, Tuple

from .models import Epistemic, Nugget, Rank, Status
from .schema import Schema


def char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Compute character n-grams of normalized string."""
    clean = "".join(c.lower() for c in text if c.isalnum() or c.isspace()).strip()
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i : i + n] for i in range(len(clean) - n + 1)}


def jaccard_value(n1: Nugget, n2: Nugget, n: int = 3) -> float:
    """Jaccard similarity over character n-grams of the object values."""
    s1 = char_ngrams(n1.fact.object, n)
    s2 = char_ngrams(n2.fact.object, n)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union if union > 0 else 0.0


class ConflictDetector:
    """Implements Algorithm 2 and manages index updates with version lineage."""

    def __init__(self, schema: Optional[Schema] = None):
        self.schema = schema or Schema()

    def integrate(self, n_new: Nugget, index_by_key: Dict[Tuple[str, str, str], List[Nugget]]) -> Nugget:
        """Execute Algorithm 2 on candidate nugget n_new against existing index pool.
        
        Args:
            n_new: Candidate nugget to integrate.
            index_by_key: Mapping from key κ = (s_norm, p, scope) to list of existing Nuggets.
        
        Returns:
            The integrated nugget (either n_new or the updated duplicate n).
        """
        # Line 1: κ <- ComputeKey(n_new)
        k = n_new.key
        
        # Line 2: N_κ <- { n in N | κ_n = κ }
        n_k = index_by_key.setdefault(k, [])

        # Line 3-4: If new key
        if not n_k:
            n_new.status = Status.ACTIVE
            if n_new.num_sources >= 2:
                n_new.epistemic.rank = Rank.PREFERRED
            n_k.append(n_new)
            return n_new

        # Line 5-7: Exact key match deduplication
        for n in n_k:
            sim = jaccard_value(n, n_new)
            if sim >= 0.85:
                # Merge evidence
                for ev in n_new.provenance.evidence:
                    n.provenance.add_evidence(ev)
                if n.num_sources >= 2 and n.status == Status.ACTIVE:
                    n.epistemic.rank = Rank.PREFERRED
                
                # Check if newly accumulated evidence resolves any existing contested state
                self._check_contested_resolution(n, n_k)
                return n

        # Line 8: r <- GetRelation(κ)
        is_multi = self.schema.is_multi_valued(n_new.fact.predicate)

        # Line 9: N_cap <- { n in N_κ | σ_n ∩ σ_{n_new} ≠ ∅ }
        n_cap = [n for n in n_k if n.validity.overlaps(n_new.validity)]

        # Line 10-11: succession (no overlap) or multi-valued relation
        if not n_cap or is_multi:
            n_new.status = Status.ACTIVE
            if n_new.num_sources >= 2:
                n_new.epistemic.rank = Rank.PREFERRED
            n_k.append(n_new)
            return n_new

        # Line 12-18: Functional relation with temporal overlap -> conflict handling
        for n in n_cap:
            t_s_new = n_new.validity.t_s
            t_s_old = n.validity.t_s

            # Branch 1 (Line 13-14): n_new has >= 2 sources and is newer
            # Succession transition: newer corroborated fact closes older fact's interval
            if n_new.num_sources >= 2 and (t_s_new and t_s_old and t_s_new > t_s_old):
                n.validity.t_e = t_s_new
                n.status = Status.ACTIVE  # Valid for historical interval [t_s_old, t_s_new)
                n_new.status = Status.ACTIVE
                n_new.provenance.parent_id = n.id

            # Uncorroborated intruder claim contradicts established consensus (Section 3.2.2)
            elif n.num_sources >= 2 and n_new.num_sources < 2:
                n_new.status = Status.DEPRECATED
                n_new.epistemic.rank = Rank.DEPRECATED
                n_new.provenance.parent_id = n.id

            # Branch 2 (Line 15-16): existing n has >= 2 sources and is newer
            elif n.num_sources >= 2 and (t_s_old and t_s_new and t_s_old > t_s_new):
                n_new.status = Status.DEPRECATED
                n_new.validity.t_e = t_s_old
                n_new.provenance.parent_id = n.id

            # Branch 3 (Line 17-18): ambiguous / equal uncorroborated sources -> Contested
            else:
                n.status = Status.CONTESTED
                n_new.status = Status.CONTESTED
                n.epistemic.rank = Rank.NORMAL
                n_new.epistemic.rank = Rank.NORMAL

        n_k.append(n_new)
        return n_new

    def _check_contested_resolution(self, n_target: Nugget, siblings: List[Nugget]) -> None:
        """Section 3.2.2: Contested resolution requiring >= 3 independent sources."""
        if n_target.status != Status.CONTESTED:
            return
        if n_target.num_sources >= 3:
            # Check other contested siblings
            resolved = False
            for sib in siblings:
                if sib.id != n_target.id and sib.status == Status.CONTESTED:
                    if n_target.num_sources > sib.num_sources:
                        sib.status = Status.DEPRECATED
                        sib.validity.t_e = n_target.validity.t_s
                        resolved = True
            if resolved:
                n_target.status = Status.ACTIVE
                n_target.epistemic.rank = Rank.PREFERRED
