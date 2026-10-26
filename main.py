"""NuggetIndex CLI & Demo Entrypoint.

Usage:
    python main.py build [--corpus data/corpus.json] [--schema data/schema.json]
    python main.py query "Who is the CEO of Acme Robotics?" --time 2020-03-01 [--view active|full] [--mode hybrid|lexical]
    python main.py answer "Who is the CEO of Acme Robotics?" --time 2020-03-01
    python main.py show-nuggets [--key <s|p|scope>]
    python main.py demo
    python main.py experiments
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from nuggetindex.llm import LLMClient
from nuggetindex.models import Status
from nuggetindex.pipeline import NuggetIndex
from nuggetindex.schema import Schema


def cmd_build(args: argparse.Namespace) -> None:
    print(f"Loading schema from {args.schema}...")
    schema = Schema.load(args.schema)

    print(f"Loading corpus from {args.corpus}...")
    with open(args.corpus, "r", encoding="utf-8") as f:
        data = json.load(f)
    documents = data.get("documents", [])
    revisions = data.get("revisions", {})

    print(f"Initializing NuggetIndex (SQLite Document Store + B-tree Metadata Index)...")
    idx = NuggetIndex(db_path=args.db, schema=schema)
    stats = idx.build(documents, revisions=revisions)

    print("\n--- Ingestion & Indexing Summary ---")
    print(f"Total documents ingested:     {stats['total_documents']}")
    print(f"Raw atomic candidates:        {stats['raw_candidates_extracted']}")
    print(f"Governed nuggets indexed:     {stats['nuggets_indexed']}")
    print(f"  Active nuggets:             {stats['active_nuggets']}")
    print(f"  Deprecated nuggets:         {stats['deprecated_nuggets']}")
    print(f"  Contested nuggets:          {stats['contested_nuggets']}")
    print(f"Database saved to {args.db}\n")


def cmd_query(args: argparse.Namespace) -> None:
    schema = Schema.load(args.schema) if Path(args.schema).exists() else None
    idx = NuggetIndex(db_path=args.db, schema=schema)

    q_time = args.time or date.today().isoformat()
    results = idx.query(
        query=args.query,
        query_time=q_time,
        view=args.view,
        mode=args.mode,
        top_k=args.top_k,
    )

    print(f"\nQuery: '{args.query}' at time={q_time} (view={args.view}, mode={args.mode})")
    print(f"Retrieved {len(results)} valid nugget(s):\n")
    for i, (nug, score) in enumerate(results, 1):
        ts = nug.validity.t_s.isoformat() if nug.validity.t_s else "-inf"
        te = nug.validity.t_e.isoformat() if nug.validity.t_e else "+inf"
        print(f"[{i}] Score={score:.4f} | Status={nug.status.value} | Validity=[{ts}, {te})")
        print(f"    Fact: ({nug.fact.subject}, {nug.fact.predicate}, {nug.fact.object})")
        print(f"    Text: \"{nug.text}\"")
        print(f"    Sources ({nug.num_sources}): {list(nug.provenance.independent_sources())}")
        print()


def cmd_answer(args: argparse.Namespace) -> None:
    schema = Schema.load(args.schema) if Path(args.schema).exists() else None
    idx = NuggetIndex(db_path=args.db, schema=schema)

    q_time = args.time or date.today().isoformat()
    resp = idx.answer(
        query=args.query,
        query_time=q_time,
        view=args.view,
        mode=args.mode,
    )

    print(f"\n================ Question & Answer ================")
    print(f"Query:      {resp['query']}")
    print(f"Query Time: {resp['query_time']}")
    print(f"\n--- Governed Evidence Injected to Generator ---\n{resp['evidence_prompt']}")
    print(f"\n--- Generator Output ---\n{resp['answer']}\n===================================================")


def cmd_show_nuggets(args: argparse.Namespace) -> None:
    idx = NuggetIndex(db_path=args.db)
    all_nugs = idx.store.get_all_nuggets()
    if args.key:
        all_nugs = [n for n in all_nugs if args.key.lower() in n.key_string.lower()]

    print(f"\nTotal nuggets: {len(all_nugs)}")
    for n in sorted(all_nugs, key=lambda x: (x.key_string, str(x.validity.t_s))):
        ts = n.validity.t_s.isoformat() if n.validity.t_s else "-inf"
        te = n.validity.t_e.isoformat() if n.validity.t_e else "+inf"
        print(f"ID={n.id} | Key={n.key_string} | Status={n.status.value:<10} | [{ts}, {te}) | Sources={n.num_sources}")
        print(f"    -> Object: {n.fact.object} | Text: \"{n.text}\"")


def cmd_demo(args: argparse.Namespace) -> None:
    """Run interactive walkthrough demonstrating all features."""
    print("=" * 70)
    print("           NUGGETINDEX: GOVERNED ATOMIC RETRIEVAL DEMO")
    print("=" * 70)

    # 1. Build
    print("\n[Step 1] Building NuggetIndex on multi-temporal corpus...")
    args.corpus = "data/corpus.json"
    args.schema = "data/schema.json"
    args.db = "nuggets.db"
    cmd_build(args)

    idx = NuggetIndex(db_path=args.db, schema=Schema.load(args.schema))

    # 2. CEO Succession queries at 3 distinct time points
    scenarios = [
        ("Who is the CEO of Acme Robotics?", "2016-05-01", "Expected: Alice Smith (2015-2019)"),
        ("Who is the CEO of Acme Robotics?", "2020-05-01", "Expected: Bob Johnson (2019-2023)"),
        ("Who is the CEO of Acme Robotics?", "2024-05-01", "Expected: Carol Davis (2023+)"),
    ]

    print("\n[Step 2] Testing CEO Succession across time periods:")
    for query, t, note in scenarios:
        print(f"\n>>> Query: '{query}' at t={t} ({note})")
        res = idx.query(query, t, view="active", top_k=3)
        for nug, sc in res:
            ts = nug.validity.t_s.isoformat() if nug.validity.t_s else "-inf"
            te = nug.validity.t_e.isoformat() if nug.validity.t_e else "+inf"
            print(f"    -> [{nug.status.value}] ({ts} to {te}): {nug.fact.object} (score={sc:.4f})")

    # 3. Contested state demonstration
    print("\n" + "=" * 70)
    print("[Step 3] Handling Source Contestation (Seattle vs Austin for Zentech HQ):")
    q_zen = "Where is the headquarters of Zentech located?"
    t_zen = "2020-07-01"

    print("\n--- (a) Querying with view='active' (filters out disputed claims) ---")
    res_act = idx.query(q_zen, t_zen, view="active", top_k=3)
    print(f"Retrieved {len(res_act)} active nuggets (disputed nuggets correctly excluded).")

    print("\n--- (b) Querying with view='full' (preserves uncertainty structurally) ---")
    res_full = idx.query(q_zen, t_zen, view="full", top_k=3)
    print(f"Retrieved {len(res_full)} nuggets:")
    for nug, sc in res_full:
        print(f"    -> [{nug.status.value}] {nug.fact.object} | sources: {list(nug.provenance.independent_sources())}")

    print("\n--- (c) Structured Prompt Generated for LLM Generator ---")
    prompt = idx.retriever.build_prompt(res_full)
    print(prompt)

    # 4. Multi-valued relation demonstration
    print("\n" + "=" * 70)
    print("[Step 4] Multi-Valued Relation (Acme Robotics Board Members):")
    q_board = "Who are the board members of Acme Robotics?"
    t_board = "2022-01-01"
    res_board = idx.query(q_board, t_board, view="active", top_k=5)
    print(f"Retrieved {len(res_board)} concurrent board members:")
    for nug, sc in res_board:
        print(f"    -> [{nug.status.value}] {nug.fact.object} (valid since {nug.validity.t_s})")

    # 5. Wikipedia Revision History Mayor update
    print("\n" + "=" * 70)
    print("[Step 5] Wikipedia Revision History Tightening (City of Lumina Mayor):")
    for q_t in ["2016-01-01", "2019-01-01", "2023-01-01"]:
        r = idx.query("Who is the mayor of Lumina?", q_t, view="active", top_k=2)
        top = r[0][0] if r else None
        if top:
            print(f"  At {q_t}: Mayor is {top.fact.object} [{top.validity.t_s} to {top.validity.t_e})")

    print("\n" + "=" * 70)
    print("                    DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="NuggetIndex: Governed Atomic Retrieval for Maintainable RAG")
    parser.add_argument("--db", default="nuggets.db", help="Path to SQLite database")
    parser.add_argument("--schema", default="data/schema.json", help="Path to schema JSON")
    parser.add_argument("--corpus", default="data/corpus.json", help="Path to corpus JSON")

    subparsers = parser.add_subparsers(dest="command")

    # build
    subparsers.add_parser("build", help="Build index from corpus and schema")

    # query
    p_query = subparsers.add_parser("query", help="Run governed retrieval query")
    p_query.add_argument("query", help="Query string")
    p_query.add_argument("--time", help="Query time (YYYY-MM-DD)")
    p_query.add_argument("--view", default="active", choices=["active", "full", "all"])
    p_query.add_argument("--mode", default="hybrid", choices=["hybrid", "lexical", "dense"])
    p_query.add_argument("--top-k", type=int, default=10)

    # answer
    p_ans = subparsers.add_parser("answer", help="Run end-to-end question answering with LLM")
    p_ans.add_argument("query", help="Query string")
    p_ans.add_argument("--time", help="Query time (YYYY-MM-DD)")
    p_ans.add_argument("--view", default="active", choices=["active", "full", "all"])
    p_ans.add_argument("--mode", default="hybrid", choices=["hybrid", "lexical", "dense"])

    # show-nuggets
    p_show = subparsers.add_parser("show-nuggets", help="Inspect indexed nuggets")
    p_show.add_argument("--key", help="Filter by key substring")

    # demo
    subparsers.add_parser("demo", help="Run complete scripted walkthrough")

    # experiments
    subparsers.add_parser("experiments", help="Run benchmark evaluation suite (RQ1-RQ5)")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "answer":
        cmd_answer(args)
    elif args.command == "show-nuggets":
        cmd_show_nuggets(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "experiments":
        from experiments.run_experiments import run_all_experiments
        run_all_experiments(db_path=args.db, schema_path=args.schema, queries_path="data/queries.json")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
