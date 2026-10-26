"""Document Store and B-Tree Metadata Index backed by SQLite.

Paper reference: Section 3.3 (Indexing and Retrieval, Components 1 & 2).
- Document Store: Key-value store mapping nugget ID to full JSON record.
- Metadata Index: B-Tree index over (status, t_s, t_e, scope) for rapid view filtering prior to ranking.
- Passages Table: Stores document chunks for baseline comparison.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .models import Nugget, Status, format_iso_date, parse_iso_date


class NuggetStore:
    """Manages SQLite storage for nuggets, metadata B-tree indexes, and passage baselines."""

    def __init__(self, db_path: str | Path = "nuggets.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            # 1. Document Store: Full nugget records
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nuggets (
                    id TEXT PRIMARY KEY,
                    key_str TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    data_json TEXT NOT NULL
                )
                """
            )

            # 2. Metadata Index: B-tree on status, t_s, t_e
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    rank TEXT NOT NULL,
                    t_s TEXT,
                    t_e TEXT,
                    scope TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    num_sources INTEGER NOT NULL,
                    FOREIGN KEY(id) REFERENCES nuggets(id) ON DELETE CASCADE
                )
                """
            )
            # Create composite B-tree index for view filtering
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_meta_status_time
                ON meta (status, t_s, t_e)
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_nuggets_key
                ON nuggets (key_str)
                """
            )

            # 3. Passage Store for baselines
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS passages (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    doc_time TEXT,
                    source_id TEXT
                )
                """
            )

    def insert_nugget(self, nugget: Nugget) -> None:
        """Insert or replace a nugget and its B-tree metadata index entry."""
        n_id = nugget.id
        n_json = json.dumps(nugget.to_dict(), ensure_ascii=False)
        ts_str = format_iso_date(nugget.validity.t_s)
        te_str = format_iso_date(nugget.validity.t_e)

        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO nuggets (id, key_str, subject, predicate, object, kind, text, data_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    n_id,
                    nugget.key_string,
                    nugget.fact.subject,
                    nugget.fact.predicate,
                    nugget.fact.object,
                    nugget.kind.value,
                    nugget.text,
                    n_json,
                ),
            )
            self.conn.execute(
                """
                INSERT OR REPLACE INTO meta (id, status, rank, t_s, t_e, scope, confidence, num_sources)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    n_id,
                    nugget.status.value,
                    nugget.epistemic.rank.value,
                    ts_str,
                    te_str,
                    nugget.validity.scope,
                    nugget.epistemic.confidence,
                    nugget.num_sources,
                ),
            )

    def insert_passage(
        self,
        passage_id: str,
        doc_id: str,
        text: str,
        doc_time: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO passages (id, doc_id, text, doc_time, source_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (passage_id, doc_id, text, doc_time, source_id or doc_id),
            )

    def get_nugget(self, nugget_id: str) -> Optional[Nugget]:
        cur = self.conn.execute("SELECT data_json FROM nuggets WHERE id = ?", (nugget_id,))
        row = cur.fetchone()
        if not row:
            return None
        return Nugget.from_dict(json.loads(row["data_json"]))

    def get_all_nuggets(self) -> List[Nugget]:
        cur = self.conn.execute("SELECT data_json FROM nuggets")
        return [Nugget.from_dict(json.loads(r["data_json"])) for r in cur.fetchall()]

    def get_all_passages(self) -> List[dict[str, Any]]:
        cur = self.conn.execute("SELECT id, doc_id, text, doc_time, source_id FROM passages")
        return [dict(r) for r in cur.fetchall()]

    def filter_ids(self, query_date: date, view: str = "active") -> Set[str]:
        """Perform B-Tree View Filtering (Section 3.3.1 Step 1).
        
        Eliminates nuggets where t not in [t_s, t_e) or where status is Deprecated.
        """
        q_str = format_iso_date(query_date)
        if view == "active":
            status_clause = "status = 'Active'"
            params = [q_str, q_str]
        elif view in ("full", "active_plus_contested"):
            status_clause = "status IN ('Active', 'Contested')"
            params = [q_str, q_str]
        elif view == "all":
            # Return all regardless of status or time
            cur = self.conn.execute("SELECT id FROM meta")
            return {r[0] for r in cur.fetchall()}
        else:
            status_clause = "status = 'Active'"
            params = [q_str, q_str]

        # SQL query using B-tree index
        query = f"""
            SELECT id FROM meta
            WHERE {status_clause}
              AND (t_s IS NULL OR t_s <= ?)
              AND (t_e IS NULL OR t_e > ?)
        """
        cur = self.conn.execute(query, params)
        return {r[0] for r in cur.fetchall()}

    def get_meta_scores(self, ids: Set[str]) -> Dict[str, float]:
        """Fetch metadata score s_meta(d) for score fusion (gamma * s_meta).
        
        Interpretation Note: s_meta combines confidence and source corroboration.
        """
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        cur = self.conn.execute(
            f"SELECT id, confidence, num_sources, rank FROM meta WHERE id IN ({placeholders})",
            list(ids),
        )
        scores = {}
        for r in cur.fetchall():
            conf = float(r["confidence"])
            src_boost = min(1.0, float(r["num_sources"]) / 3.0)
            rank_boost = 1.0 if r["rank"] == "Preferred" else 0.8
            scores[r["id"]] = conf * 0.5 + src_boost * 0.3 + rank_boost * 0.2
        return scores

    def close(self) -> None:
        self.conn.close()

