"""Baseline retrieval models for comparative evaluation.

Paper reference: Section 4.3 (Baselines).
Includes:
- BM25-Passage, Dense-Passage, Hybrid-Passage
- Time-aware passage variants: TimeFilter (±180 days), RecencyRerank (decay λ=0.001), LatestSnapshot
- Proposition-RAG (atomic propositions without governance) and Proposition + TimeFilter
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Dict, List, Optional, Set, Tuple

from .dense import DenseIndex
from .models import format_iso_date, parse_iso_date
from .retrieval import min_max_normalize
from .sparse import BM25Index


class PassageRetriever:
    """Passage-level retrieval baselines."""

    def __init__(self, passages: List[dict[str, Any]], alpha: float = 0.5, beta: float = 0.5):
        self.passages = passages
        self.alpha = alpha
        self.beta = beta
        self.sparse = BM25Index()
        self.dense = DenseIndex()
        self._build_indices()

    def _build_indices(self) -> None:
        docs = [(p["id"], p["text"]) for p in self.passages]
        self.sparse.index_documents(docs)
        self.dense.fit_and_index(docs)

    def retrieve(
        self,
        query: str,
        query_time: Optional[str | date] = None,
        strategy: str = "hybrid",  # "bm25" | "dense" | "hybrid" | "time_filter" | "recency" | "latest"
        top_k: int = 10,
        time_window_days: int = 180,
        recency_lambda: float = 0.001,
    ) -> List[Tuple[dict[str, Any], float]]:
        q_date = parse_iso_date(query_time) if isinstance(query_time, str) else query_time

        # Candidate filtering if latest snapshot or time filter
        allowed_ids: Optional[Set[str]] = None

        if strategy == "time_filter" and q_date:
            allowed_ids = set()
            for p in self.passages:
                d_time = parse_iso_date(p.get("doc_time"))
                if d_time:
                    diff = abs((q_date - d_time).days)
                    if diff <= time_window_days:
                        allowed_ids.add(p["id"])
            if not allowed_ids:
                return []

        elif strategy == "latest":
            # Group passages by doc topic / doc_id prefix and keep only latest version
            grouped: Dict[str, Tuple[date, str]] = {}
            for p in self.passages:
                d_time = parse_iso_date(p.get("doc_time")) or date.min
                doc_key = p["doc_id"].split("_v")[0]  # strip version suffix if present
                if doc_key not in grouped or d_time > grouped[doc_key][0]:
                    grouped[doc_key] = (d_time, p["id"])
            allowed_ids = {pid for _, pid in grouped.values()}

        # Scores
        lex = min_max_normalize(self.sparse.search(query, allowed_ids=allowed_ids))
        dense = min_max_normalize(self.dense.search(query, allowed_ids=allowed_ids))

        target_ids = allowed_ids if allowed_ids is not None else {p["id"] for p in self.passages}
        final_scores: Dict[str, float] = {}

        for pid in target_ids:
            s_l = lex.get(pid, 0.0)
            s_d = dense.get(pid, 0.0)
            if strategy == "bm25":
                score = s_l
            elif strategy == "dense":
                score = s_d
            else:
                score = self.alpha * s_l + self.beta * s_d

            # Apply recency decay if recency reranking is active
            if strategy == "recency" and q_date:
                p_item = next((p for p in self.passages if p["id"] == pid), None)
                if p_item and p_item.get("doc_time"):
                    d_time = parse_iso_date(p_item["doc_time"])
                    if d_time:
                        days_diff = max(0, (q_date - d_time).days)
                        score = score * math.exp(-recency_lambda * days_diff)

            final_scores[pid] = score

        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        id_map = {p["id"]: p for p in self.passages}
        return [(id_map[pid], score) for pid, score in ranked if pid in id_map]


class PropositionRetriever:
    """Proposition-RAG baseline: retrieves atomic propositions as static text without governance."""

    def __init__(self, propositions: List[dict[str, Any]]):
        # propositions: list of {id, text, doc_id, doc_time}
        self.propositions = propositions
        self.sparse = BM25Index()
        self.dense = DenseIndex()
        docs = [(p["id"], p["text"]) for p in self.propositions]
        self.sparse.index_documents(docs)
        self.dense.fit_and_index(docs)

    def retrieve(
        self,
        query: str,
        query_time: Optional[str | date] = None,
        apply_time_filter: bool = False,
        time_window_days: int = 180,
        top_k: int = 20,
    ) -> List[Tuple[dict[str, Any], float]]:
        q_date = parse_iso_date(query_time) if isinstance(query_time, str) else query_time
        allowed_ids: Optional[Set[str]] = None

        if apply_time_filter and q_date:
            allowed_ids = set()
            for p in self.propositions:
                d_time = parse_iso_date(p.get("doc_time"))
                if d_time:
                    diff = abs((q_date - d_time).days)
                    if diff <= time_window_days:
                        allowed_ids.add(p["id"])
            if not allowed_ids:
                return []

        lex = min_max_normalize(self.sparse.search(query, allowed_ids=allowed_ids))
        dense = min_max_normalize(self.dense.search(query, allowed_ids=allowed_ids))

        target_ids = allowed_ids if allowed_ids is not None else {p["id"] for p in self.propositions}
        final_scores = {}
        for pid in target_ids:
            final_scores[pid] = 0.5 * lex.get(pid, 0.0) + 0.5 * dense.get(pid, 0.0)

        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        id_map = {p["id"]: p for p in self.propositions}
        return [(id_map[pid], score) for pid, score in ranked if pid in id_map]

