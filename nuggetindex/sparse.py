"""Component 3: Sparse Inverted Index with BM25.

Paper reference: Section 3.3 (Sparse Index) and Section 3.3.1 (Candidate Retrieval).
- Inverted index over canonical nugget text.
- BM25 scoring with k1=1.2, b=0.75.
- Supports pre-filtering by allowed ID set for governed view filtering.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple


def tokenize(text: str) -> List[str]:
    """Tokenize and lower-case text."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Index:
    """Inverted index implementing BM25 with pre-filtered candidate retrieval."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lens: Dict[str, int] = {}
        self.avg_doc_len: float = 0.0
        # term -> list of (doc_id, term_frequency)
        self.inverted_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        # term -> document frequency
        self.df: Dict[str, int] = defaultdict(int)
        self.total_docs: int = 0
        self.doc_texts: Dict[str, str] = {}

    def index_documents(self, documents: List[Tuple[str, str]]) -> None:
        """Build inverted index from list of (doc_id, text)."""
        self.doc_lens.clear()
        self.inverted_index.clear()
        self.df.clear()
        self.doc_texts.clear()
        self.total_docs = len(documents)

        total_len = 0
        for doc_id, text in documents:
            tokens = tokenize(text)
            length = len(tokens)
            self.doc_lens[doc_id] = length
            self.doc_texts[doc_id] = text
            total_len += length

            counts = Counter(tokens)
            for term, freq in counts.items():
                self.inverted_index[term].append((doc_id, freq))
                self.df[term] += 1

        self.avg_doc_len = (total_len / self.total_docs) if self.total_docs > 0 else 0.0

    def idf(self, term: str) -> float:
        """BM25 Robertson-Spärck Jones IDF."""
        n_q = self.df.get(term, 0)
        if n_q == 0:
            return 0.0
        # ln( (N - n_q + 0.5) / (n_q + 0.5) + 1 )
        val = (self.total_docs - n_q + 0.5) / (n_q + 0.5) + 1.0
        return math.log(max(val, 1e-6))

    def search(
        self,
        query: str,
        allowed_ids: Optional[Set[str]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, float]:
        """Search query and return {doc_id: bm25_score} restricted to allowed_ids."""
        q_tokens = tokenize(query)
        scores: Dict[str, float] = defaultdict(float)

        for term in q_tokens:
            postings = self.inverted_index.get(term, [])
            term_idf = self.idf(term)
            if term_idf <= 0:
                continue

            for doc_id, freq in postings:
                if allowed_ids is not None and doc_id not in allowed_ids:
                    continue
                d_len = self.doc_lens.get(doc_id, 0)
                denom = freq + self.k1 * (1.0 - self.b + self.b * (d_len / (self.avg_doc_len or 1.0)))
                scores[doc_id] += term_idf * (freq * (self.k1 + 1.0)) / (denom or 1.0)

        # Ensure all allowed IDs get a score entry (at least 0.0) if requested
        if allowed_ids is not None:
            for aid in allowed_ids:
                if aid not in scores:
                    scores[aid] = 0.0

        if top_k is not None:
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            return dict(sorted_items)
        return dict(scores)

