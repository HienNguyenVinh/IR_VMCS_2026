"""Unit tests for B-Tree View Filtering, Hybrid Score Fusion, and Prompt Construction."""

from datetime import date
from nuggetindex.dense import DenseIndex
from nuggetindex.models import Epistemic, Evidence, Fact, Kind, Nugget, Provenance, Rank, Status, Validity
from nuggetindex.retrieval import NuggetRetriever
from nuggetindex.sparse import BM25Index
from nuggetindex.store import NuggetStore


def make_stored_nugget(store: NuggetStore, subject: str, predicate: str, obj: str, t_s: str, t_e: str = None, status: Status = Status.ACTIVE) -> Nugget:
    text = f"{subject} {predicate} {obj}"
    val = Validity(t_s=date.fromisoformat(t_s), t_e=date.fromisoformat(t_e) if t_e else None)
    nug = Nugget(
        kind=Kind.SEMANTIC_FACT,
        fact=Fact(subject, predicate, obj),
        validity=val,
        epistemic=Epistemic(status=status),
        provenance=Provenance(evidence=[Evidence("d1", "s1", 0, len(text), text)]),
        text=text,
        normalized_subject=subject.lower(),
    )
    store.insert_nugget(nug)
    return nug


def test_view_filtering_excludes_deprecated_and_out_of_interval(tmp_path):
    db_file = tmp_path / "test.db"
    store = NuggetStore(db_file)

    # Nugget 1: Alice (2015-2019, Active)
    n1 = make_stored_nugget(store, "Acme", "chiefExecutiveOfficer", "Alice Smith", "2015-01-01", "2019-01-01", Status.ACTIVE)
    # Nugget 2: Bob (2019+, Active)
    n2 = make_stored_nugget(store, "Acme", "chiefExecutiveOfficer", "Bob Johnson", "2019-01-01", None, Status.ACTIVE)
    # Nugget 3: Dan (Deprecated)
    n3 = make_stored_nugget(store, "Acme", "chiefExecutiveOfficer", "Dan Pham", "2018-01-01", None, Status.DEPRECATED)

    # Query at 2017: only Alice is valid
    ids_2017 = store.filter_ids(date(2017, 6, 1), view="active")
    assert n1.id in ids_2017
    assert n2.id not in ids_2017
    assert n3.id not in ids_2017

    # Query at 2021: only Bob is valid
    ids_2021 = store.filter_ids(date(2021, 6, 1), view="active")
    assert n1.id not in ids_2021
    assert n2.id in ids_2021
    assert n3.id not in ids_2021

    store.close()


def test_contested_view_flagging(tmp_path):
    db_file = tmp_path / "test2.db"
    store = NuggetStore(db_file)

    n_cont = make_stored_nugget(store, "Zentech", "headquarters", "Seattle", "2020-01-01", None, Status.CONTESTED)
    n_act = make_stored_nugget(store, "Acme", "headquarters", "Boston", "2015-01-01", None, Status.ACTIVE)

    # Active view should not return contested
    act_ids = store.filter_ids(date(2020, 6, 1), view="active")
    assert n_cont.id not in act_ids
    assert n_act.id in act_ids

    # Full view should return contested
    full_ids = store.filter_ids(date(2020, 6, 1), view="full")
    assert n_cont.id in full_ids
    assert n_act.id in full_ids

    # Test retriever prompt formatting
    sparse = BM25Index()
    dense = DenseIndex()
    retriever = NuggetRetriever(store, sparse, dense)
    results = [(n_act, 0.95), (n_cont, 0.85)]
    prompt = retriever.build_prompt(results)

    assert "Established facts:" in prompt
    assert "Disputed (sources disagree):" in prompt
    assert "Seattle" in prompt

    store.close()

