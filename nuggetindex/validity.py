"""Algorithm 1: Validity Interval Inference.

Paper reference: Section 3.2.1 (Stage 3: Validity Inference, Algorithm 1).
- Extracts temporal expressions (START, END, POINT) from the nugget text.
- Anchors start date t_s to explicit start/point, or defaults to doc timestamp t_d.
- Anchors end date t_e to explicit end, or defaults to +infinity.
- Tightens [t_s, t_e) using document revision history H.
- Tightens t_e using temporal succession from sibling nuggets with >= 2 sources.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from .models import Nugget, Validity, parse_iso_date
from .temporal import TempType, extract_temporal_expressions


def earlier(d1: Optional[date], d2: Optional[date]) -> Optional[date]:
    """Earlier of two dates where None denotes +infinity."""
    if d1 is None:
        return d2
    if d2 is None:
        return d1
    return min(d1, d2)


def later(d1: Optional[date], d2: Optional[date]) -> Optional[date]:
    """Later of two dates."""
    if d1 is None:
        return d2
    if d2 is None:
        return d1
    return max(d1, d2)


class RevisionDoc:
    """A versioned document revision with timestamp and text content."""

    def __init__(self, rev_id: str, timestamp: str | date, text: str):
        self.rev_id = rev_id
        self.time = parse_iso_date(timestamp) if isinstance(timestamp, str) else timestamp
        self.text = text


def infer_validity(
    nugget: Nugget,
    doc_time: Optional[str | date] = None,
    revision_history: Optional[List[RevisionDoc]] = None,
    sibling_nuggets: Optional[List[Nugget]] = None,
) -> Validity:
    """Implement Algorithm 1: Validity Interval Inference.
    
    Args:
        nugget: The candidate nugget.
        doc_time: Document timestamp t_d.
        revision_history: Time-ordered list of revisions for this document.
        sibling_nuggets: Other extracted nuggets sharing the same key κ.
    """
    text = nugget.text
    t_d = parse_iso_date(doc_time) if isinstance(doc_time, str) else doc_time

    # Line 1: T <- ExtractTemporalExpressions(l_n)
    T = extract_temporal_expressions(text)

    # Line 2: t_s selection
    start_expr = next((e for e in T if e.expr_type == TempType.START), None)
    point_expr = next((e for e in T if e.expr_type == TempType.POINT), None)

    inferred = False
    source_type = "doc_default"

    if start_expr:
        t_s = start_expr.date_val
        source_type = "explicit"
    elif point_expr:
        t_s = point_expr.date_val
        source_type = "explicit"
    elif t_d is not None:
        t_s = t_d
        inferred = True
        source_type = "inferred"
    else:
        # If neither explicit date nor doc_time is provided, fallback to min date
        t_s = date(2000, 1, 1)
        inferred = True
        source_type = "default"

    # Line 3: t_e selection
    end_expr = next((e for e in T if e.expr_type == TempType.END), None)
    if end_expr:
        t_e = end_expr.date_val
        source_type = "explicit"
    else:
        t_e = None  # None represents +infinity

    # Lines 4-11: Revision history tightening
    if revision_history and len(revision_history) > 1:
        # Ensure time-ordered
        R = sorted(revision_history, key=lambda r: r.time or date.min)
        for i in range(len(R) - 1):
            r_curr, r_next = R[i], R[i + 1]
            t_next = r_next.time
            # Check fact occurrence using normalized text or object
            in_curr = _fact_in_text(nugget, r_curr.text)
            in_next = _fact_in_text(nugget, r_next.text)

            # Line 8-9: absent in next revision -> fact superseded/deleted
            if in_curr and not in_next and t_next:
                t_e = earlier(t_e, t_next)
                source_type = "revision"

            # Line 10-11: first appears in next revision -> fact started
            if not in_curr and in_next and t_next:
                t_s = later(t_s, t_next)
                source_type = "revision"

    # Lines 12-15: Tighten via conflicts / succession
    if sibling_nuggets:
        for n_prime in sibling_nuggets:
            if n_prime.key == nugget.key and n_prime.fact.object.lower() != nugget.fact.object.lower():
                n_prime_ts = n_prime.validity.t_s
                if n_prime_ts and t_s and n_prime_ts > t_s and n_prime.num_sources >= 2:
                    t_e = earlier(t_e, n_prime_ts)
                    source_type = "conflict"

    validity = Validity(
        t_s=t_s,
        t_e=t_e,
        scope=nugget.validity.scope,
        location=nugget.validity.location,
        source_type=source_type,
        inferred=inferred,
    )
    nugget.validity = validity
    return validity


def _fact_in_text(nugget: Nugget, text: str) -> bool:
    """Check whether the fact is asserted in the text."""
    n_lower = nugget.text.lower()
    t_lower = text.lower()
    if n_lower in t_lower:
        return True
    # Check subject and object co-occurrence
    s_lower = nugget.fact.subject.lower()
    o_lower = nugget.fact.object.lower()
    if s_lower in t_lower and o_lower in t_lower:
        return True
    return False

