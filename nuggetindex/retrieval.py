"""Governed Retrieval Pipeline and Prompt Formatter.

Paper reference: Section 3.3.1 (Retrieval Algorithm) and Section 3.3.2 (Handling Contested Nuggets).
Four-step procedure:
1. View filtering: restricts search to nuggets in target view valid at query time t.
2. Candidate retrieval: BM25 sparse scores + HNSW dense scores on the filtered set.
3. Score fusion: s(x, n) = α·s_lex + β·s_dense + γ·s_meta with min-max normalization.
4. Ranking: return top-K results.

Prompt generation:
Formats retrieved evidence into structured "Established facts" vs "Disputed (sources disagree)".
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Set, Tuple

from .dense import DenseIndex
from .models import Nugget, Status, parse_iso_date
from .sparse import BM25Index
from .store import NuggetStore


def min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize score dictionary to [0, 1]."""
    if not scores:
        return {}
    vals = list(scores.values())
    min_v = min(vals)
    max_v = max(vals)
    if max_v - min_v < 1e-9:
        return {k: 1.0 if max_v > 0 else 0.0 for k in scores}
    return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}


class NuggetRetriever:
    """Orchestrates view-filtered hybrid and lexical-only nugget retrieval."""

    def __init__(
        self,
        store: NuggetStore,
        sparse_index: BM25Index,
        dense_index: DenseIndex,
        alpha: float = 0.4,
        beta: float = 0.5,
        gamma: float = 0.1,
    ):
        self.store = store
        self.sparse = sparse_index
        self.dense = dense_index
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def retrieve(
        self,
        query: str,
        query_time: str | date,
        view: str = "active",  # "active" | "full" / "active_plus_contested" | "all"
        mode: str = "hybrid",  # "hybrid" | "lexical" | "dense"
        top_k: int = 20,
    ) -> List[Tuple[Nugget, float]]:
        """Execute 4-step governed retrieval."""
        q_date = parse_iso_date(query_time) if isinstance(query_time, str) else query_time
        if q_date is None:
            raise ValueError(f"Invalid query date: {query_time}")

        # Step 1: View Filtering
        allowed_ids = self.store.filter_ids(q_date, view=view)
        if not allowed_ids:
            return []

        # Step 2: Candidate Retrieval
        lex_raw = self.sparse.search(query, allowed_ids=allowed_ids)
        lex_norm = min_max_normalize(lex_raw)

        if mode in ("hybrid", "dense"):
            dense_raw = self.dense.search(query, allowed_ids=allowed_ids)
            dense_norm = min_max_normalize(dense_raw)
        else:
            dense_norm = {aid: 0.0 for aid in allowed_ids}

        # Metadata scores
        meta_raw = self.store.get_meta_scores(allowed_ids)
        meta_norm = min_max_normalize(meta_raw)

        # Step 3: Score Fusion
        # Modes:
        # - hybrid: α·lex + β·dense + γ·meta (Section 4.4: 0.4 / 0.5 / 0.1)
        # - lexical: 1.0·lex + 0.1·meta
        # - dense: 1.0·dense + 0.1·meta
        combined_scores: Dict[str, float] = {}
        for nid in allowed_ids:
            s_lex = lex_norm.get(nid, 0.0)
            s_dense = dense_norm.get(nid, 0.0)
            s_meta = meta_norm.get(nid, 0.0)

            if mode == "hybrid":
                total = self.alpha * s_lex + self.beta * s_dense + self.gamma * s_meta
            elif mode == "lexical":
                total = 0.9 * s_lex + 0.1 * s_meta
            elif mode == "dense":
                total = 0.9 * s_dense + 0.1 * s_meta
            else:
                total = self.alpha * s_lex + self.beta * s_dense + self.gamma * s_meta
            combined_scores[nid] = total

        # Step 4: Top-K Ranking
        ranked_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for nid, score in ranked_ids:
            nug = self.store.get_nugget(nid)
            if nug:
                results.append((nug, score))

        return results

    def build_prompt(self, results: List[Tuple[Nugget, float]]) -> str:
        """Section 3.3.2: Format retrieved nuggets into Established vs Disputed facts."""
        active_facts = []
        disputed_facts = []

        for nug, _ in results:
            if nug.status == Status.ACTIVE:
                active_facts.append(f"- {nug.text}")
            elif nug.status == Status.CONTESTED:
                # Group evidence by source
                sources_str = ", ".join(
                    f"{ev.source_id}: '{ev.text}'" for ev in nug.provenance.evidence
                )
                disputed_facts.append(f"- [{nug.text}]: {sources_str}")

        lines = ["Established facts:"]
        if active_facts:
            lines.extend(active_facts)
        else:
            lines.append("  (None found)")

        if disputed_facts:
            lines.append("\nDisputed (sources disagree):")
            lines.extend(disputed_facts)

        return "\n".join(lines)

