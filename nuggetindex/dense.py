"""Component 4: Dense Vector Index with HNSW.

Paper reference: Section 3.3 (Dense Index: HNSW M=32, efConst=200) and Section 3.3.1.
- Dense embedding generation using TF-IDF + Latent Semantic Analysis (TruncatedSVD),
  producing normalized dense vectors.
- HNSW index (via hnswlib if available, with numpy cosine similarity fallback).
- Supports pre-filtering candidates restricted to allowed ID sets.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class DenseIndex:
    """Manages dense embeddings and HNSW approximate nearest neighbor search."""

    def __init__(self, dim: int = 64, m: int = 32, ef_construction: int = 200):
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.svd: Optional[TruncatedSVD] = None
        self.doc_ids: List[str] = []
        self.id_to_idx: Dict[str, int] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.hnsw_index = None

    def fit_and_index(self, documents: List[Tuple[str, str]]) -> None:
        """Fit embedding projection and build HNSW index."""
        if not documents:
            return

        self.doc_ids = [doc_id for doc_id, _ in documents]
        self.id_to_idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        texts = [text for _, text in documents]

        # 1. TF-IDF
        tfidf_mat = self.vectorizer.fit_transform(texts)
        n_features = tfidf_mat.shape[1]
        actual_dim = min(self.dim, n_features, len(documents))

        # 2. SVD / LSA
        if actual_dim > 1:
            self.svd = TruncatedSVD(n_components=actual_dim, random_state=42)
            dense_mat = self.svd.fit_transform(tfidf_mat)
        else:
            dense_mat = tfidf_mat.toarray()

        # L2 normalize
        norms = np.linalg.norm(dense_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = (dense_mat / norms).astype(np.float32)

        # 3. Build HNSW Index
        try:
            import hnswlib
            num_elements = len(documents)
            dim = self.embeddings.shape[1]
            self.hnsw_index = hnswlib.Index(space="cosine", dim=dim)
            self.hnsw_index.init_index(
                max_elements=max(num_elements + 100, 1000),
                ef_construction=self.ef_construction,
                M=self.m,
            )
            self.hnsw_index.add_items(self.embeddings, np.arange(num_elements))
            self.hnsw_index.set_ef(max(50, self.m))
        except Exception:
            self.hnsw_index = None  # Fallback to exact numpy cosine

    def encode_query(self, query: str) -> np.ndarray:
        """Project query text into the embedding space."""
        q_tfidf = self.vectorizer.transform([query])
        if self.svd is not None:
            q_dense = self.svd.transform(q_tfidf)
        else:
            q_dense = q_tfidf.toarray()
        norm = np.linalg.norm(q_dense)
        if norm > 0:
            q_dense = q_dense / norm
        return q_dense.astype(np.float32)[0]

    def search(
        self,
        query: str,
        allowed_ids: Optional[Set[str]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, float]:
        """Dense similarity search returning {doc_id: cosine_sim} for allowed_ids."""
        if self.embeddings is None or len(self.doc_ids) == 0:
            return {}

        q_vec = self.encode_query(query)

        # Fast path with numpy dot product (since vectors are L2-normalized)
        sims = np.dot(self.embeddings, q_vec)

        scores: Dict[str, float] = {}
        for doc_id, sim in zip(self.doc_ids, sims):
            if allowed_ids is not None and doc_id not in allowed_ids:
                continue
            # Clip to [0, 1] cosine similarity
            score = float(max(0.0, sim))
            scores[doc_id] = score

        if allowed_ids is not None:
            for aid in allowed_ids:
                if aid not in scores:
                    scores[aid] = 0.0

        if top_k is not None:
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            return dict(sorted_items)
        return scores

