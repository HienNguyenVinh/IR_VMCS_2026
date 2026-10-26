"""Main NuggetIndex Pipeline orchestrator.

Integrates Stages 1–4, indexing, governed retrieval, and generation.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .baselines import PassageRetriever, PropositionRetriever
from .canonicalize import Canonicalizer
from .conflict import ConflictDetector, jaccard_value
from .dense import DenseIndex
from .extraction import CandidateExtractor, RawCandidate
from .llm import LLMClient
from .models import Nugget, Status, format_iso_date, parse_iso_date
from .retrieval import NuggetRetriever
from .schema import Schema
from .sparse import BM25Index
from .store import NuggetStore
from .validity import RevisionDoc, infer_validity


class NuggetIndex:
    """Governed Atomic Retrieval for Maintainable RAG."""

    def __init__(
        self,
        db_path: str | Path = "nuggets.db",
        schema: Optional[Schema] = None,
        llm_client: Optional[LLMClient] = None,
        alpha: float = 0.4,
        beta: float = 0.5,
        gamma: float = 0.1,
    ):
        self.store = NuggetStore(db_path)
        self.schema = schema or Schema()
        self.llm = llm_client or LLMClient()
        self.extractor = CandidateExtractor(self.llm)
        self.canonicalizer = Canonicalizer(self.schema)
        self.conflict_detector = ConflictDetector(self.schema)

        self.sparse_index = BM25Index()
        self.dense_index = DenseIndex()
        self.retriever = NuggetRetriever(
            self.store,
            self.sparse_index,
            self.dense_index,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        self.passage_retriever: Optional[PassageRetriever] = None
        self.prop_retriever: Optional[PropositionRetriever] = None

        # Auto-reload indexes from store if database already populated
        existing_nuggets = self.store.get_all_nuggets()
        if existing_nuggets:
            doc_tuples = [(n.id, n.text) for n in existing_nuggets]
            self.sparse_index.index_documents(doc_tuples)
            self.dense_index.fit_and_index(doc_tuples)

        existing_passages = self.store.get_all_passages()
        if existing_passages:
            self.passage_retriever = PassageRetriever(existing_passages)

    def build(
        self,
        documents: List[dict[str, Any]],
        revisions: Optional[Dict[str, List[dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        """Build NuggetIndex from a corpus of documents with optional revision history.
        
        Args:
            documents: List of dicts with keys {id, text, doc_time, source_id, title}
            revisions: Mapping of doc_id to list of revisions {rev_id, timestamp, text}
        """
        # Parse revision history objects
        parsed_revisions: Dict[str, List[RevisionDoc]] = {}
        if revisions:
            for d_id, rev_list in revisions.items():
                parsed_revisions[d_id] = [
                    RevisionDoc(r["rev_id"], r["timestamp"], r["text"])
                    for r in rev_list
                ]

        # 1. Stage 1: Candidate Extraction
        all_raw_candidates: List[RawCandidate] = []
        for doc in documents:
            cands = self.extractor.extract_document(
                doc_id=doc["id"],
                text=doc["text"],
                doc_time=doc.get("doc_time"),
                source_id=doc.get("source_id", doc["id"]),
            )
            all_raw_candidates.extend(cands)

            # Store passages for baselines (paragraphs / chunks)
            self.store.insert_passage(
                passage_id=f"pass_{doc['id']}",
                doc_id=doc["id"],
                text=doc["text"],
                doc_time=doc.get("doc_time"),
                source_id=doc.get("source_id"),
            )

        # 2. Stage 2: Canonicalization
        initial_nuggets: List[Nugget] = [
            self.canonicalizer.canonicalize_candidate(c) for c in all_raw_candidates
        ]

        # Pre-group by key κ to pre-accumulate evidence across duplicates
        grouped_by_key: Dict[Tuple[str, str, str], List[Nugget]] = defaultdict(list)
        for nug in initial_nuggets:
            k = nug.key
            # Check near-duplicate in current group
            matched = False
            for existing in grouped_by_key[k]:
                if jaccard_value(existing, nug) >= 0.85:
                    for ev in nug.provenance.evidence:
                        existing.provenance.add_evidence(ev)
                    matched = True
                    break
            if not matched:
                grouped_by_key[k].append(nug)

        deduped_nuggets: List[Nugget] = []
        for nug_list in grouped_by_key.values():
            deduped_nuggets.extend(nug_list)

        # 3. Stage 3: Algorithm 1 (Validity Interval Inference)
        for nug in deduped_nuggets:
            doc_id = nug.provenance.evidence[0].doc_id if nug.provenance.evidence else ""
            doc_time = nug.provenance.evidence[0].doc_time if nug.provenance.evidence else None
            h_rev = parsed_revisions.get(doc_id)
            siblings = [s for s in deduped_nuggets if s.key == nug.key and s is not nug]
            infer_validity(nug, doc_time=doc_time, revision_history=h_rev, sibling_nuggets=siblings)

        # 4. Stage 4: Algorithm 2 (Conflict Detection and State Assignment)
        # Sort by start date t_s to process historically
        deduped_nuggets.sort(key=lambda n: n.validity.t_s or date.min)

        index_by_key: Dict[Tuple[str, str, str], List[Nugget]] = defaultdict(list)
        final_nuggets: List[Nugget] = []

        for nug in deduped_nuggets:
            integrated = self.conflict_detector.integrate(nug, index_by_key)
            if integrated not in final_nuggets:
                final_nuggets.append(integrated)

        # Persist into Store (SQLite Document Store + B-tree Metadata Index)
        for nug in final_nuggets:
            self.store.insert_nugget(nug)

        # 5. Build Sparse BM25 and Dense HNSW indexes
        doc_tuples = [(n.id, n.text) for n in final_nuggets]
        self.sparse_index.index_documents(doc_tuples)
        self.dense_index.fit_and_index(doc_tuples)

        # Build baseline retrievers
        passages = self.store.get_all_passages()
        self.passage_retriever = PassageRetriever(passages)

        propositions = [
            {
                "id": f"prop_{i}",
                "text": c.text,
                "doc_id": c.doc_id,
                "doc_time": c.doc_time,
            }
            for i, c in enumerate(all_raw_candidates)
        ]
        self.prop_retriever = PropositionRetriever(propositions)

        stats = {
            "total_documents": len(documents),
            "raw_candidates_extracted": len(all_raw_candidates),
            "nuggets_indexed": len(final_nuggets),
            "active_nuggets": sum(1 for n in final_nuggets if n.status == Status.ACTIVE),
            "deprecated_nuggets": sum(1 for n in final_nuggets if n.status == Status.DEPRECATED),
            "contested_nuggets": sum(1 for n in final_nuggets if n.status == Status.CONTESTED),
        }
        return stats

    def add_document(
        self,
        doc_id: str,
        text: str,
        doc_time: Optional[str] = None,
        source_id: Optional[str] = None,
        revision_history: Optional[List[RevisionDoc]] = None,
    ) -> List[Nugget]:
        """Incremental ingestion of a new document."""
        cands = self.extractor.extract_document(doc_id, text, doc_time, source_id)
        new_nuggets = []
        all_existing = self.store.get_all_nuggets()
        index_by_key = defaultdict(list)
        for n in all_existing:
            index_by_key[n.key].append(n)

        for c in cands:
            nug = self.canonicalizer.canonicalize_candidate(c)
            siblings = index_by_key.get(nug.key, [])
            infer_validity(nug, doc_time=doc_time, revision_history=revision_history, sibling_nuggets=siblings)
            integrated = self.conflict_detector.integrate(nug, index_by_key)
            self.store.insert_nugget(integrated)
            new_nuggets.append(integrated)

        # Refresh index
        all_updated = self.store.get_all_nuggets()
        doc_tuples = [(n.id, n.text) for n in all_updated]
        self.sparse_index.index_documents(doc_tuples)
        self.dense_index.fit_and_index(doc_tuples)

        return new_nuggets

    def query(
        self,
        query: str,
        query_time: str | date,
        view: str = "active",
        mode: str = "hybrid",
        top_k: int = 20,
    ) -> List[Tuple[Nugget, float]]:
        """Retrieve relevant nuggets under temporal validity governance."""
        return self.retriever.retrieve(
            query=query,
            query_time=query_time,
            view=view,
            mode=mode,
            top_k=top_k,
        )

    def answer(
        self,
        query: str,
        query_time: str | date,
        view: str = "active",
        mode: str = "hybrid",
        top_k: int = 20,
    ) -> Dict[str, Any]:
        """End-to-end RAG answer generation using structured governed evidence."""
        results = self.query(query, query_time, view=view, mode=mode, top_k=top_k)
        evidence_prompt = self.retriever.build_prompt(results)

        system_prompt = (
            "You are an accurate, time-aware question answering assistant. "
            "Use ONLY the retrieved evidence below to answer the user question. "
            "If evidence is marked 'Disputed (sources disagree)', explicitly reflect "
            "the disagreement and cite differing sources rather than picking one. "
            "Do not use outdated facts."
        )

        user_prompt = (
            f"Query time: {query_time}\n"
            f"Question: {query}\n\n"
            f"Retrieved Evidence:\n{evidence_prompt}\n\n"
            f"Answer:"
        )

        try:
            answer_text = self.llm.complete(user_prompt, system_prompt)
        except Exception as e:
            answer_text = f"[Answer generation failed: {e}]"

        return {
            "query": query,
            "query_time": str(query_time),
            "answer": answer_text,
            "evidence_prompt": evidence_prompt,
            "nuggets": [(n.to_dict(), score) for n, score in results],
        }
