"""Unit tests for Algorithm 1: Validity Interval Inference."""

from datetime import date
from nuggetindex.models import Epistemic, Evidence, Fact, Kind, Nugget, Provenance, Validity
from nuggetindex.validity import RevisionDoc, infer_validity


def make_test_nugget(text: str, subject: str, predicate: str, obj: str, num_sources: int = 1) -> Nugget:
    evs = [
        Evidence(doc_id=f"doc_{i}", source_id=f"src_{i}", start=0, end=len(text), text=text)
        for i in range(num_sources)
    ]
    return Nugget(
        kind=Kind.SEMANTIC_FACT,
        fact=Fact(subject, predicate, obj),
        validity=Validity(t_s=None, t_e=None),
        epistemic=Epistemic(),
        provenance=Provenance(evidence=evs),
        text=text,
        normalized_subject=subject.lower(),
    )


def test_default_doc_timestamp():
    nug = make_test_nugget("Acme Robotics is in Boston.", "Acme Robotics", "headquarters", "Boston")
    val = infer_validity(nug, doc_time="2016-06-01")
    assert val.t_s == date(2016, 6, 1)
    assert val.t_e is None
    assert val.inferred is True


def test_explicit_temporal_cues():
    nug = make_test_nugget(
        "Alice Smith was appointed CEO in January 2015 and served until January 2019.",
        "Acme Robotics",
        "chiefExecutiveOfficer",
        "Alice Smith",
    )
    val = infer_validity(nug, doc_time="2020-01-01")
    assert val.t_s == date(2015, 1, 1)
    assert val.t_e == date(2019, 1, 1)
    assert val.inferred is False


def test_revision_history_tightening():
    nug = make_test_nugget("Arthur Pendelton is the mayor of Lumina.", "Lumina", "mayor", "Arthur Pendelton")
    revs = [
        RevisionDoc("r1", "2015-02-01", "Arthur Pendelton is the mayor of Lumina."),
        RevisionDoc("r2", "2018-04-10", "Beatrice Sterling is the mayor of Lumina."),
    ]
    val = infer_validity(nug, doc_time="2015-02-01", revision_history=revs)
    # Absent in r2 -> t_e tightened to Time(r2) = 2018-04-10
    assert val.t_s == date(2015, 2, 1)
    assert val.t_e == date(2018, 4, 10)
    assert val.source_type == "revision"


def test_conflict_tightening():
    # Sibling with >= 2 sources and later t_s tightens older nugget's t_e
    nug_old = make_test_nugget("Alice Smith is CEO.", "Acme Robotics", "chiefExecutiveOfficer", "Alice Smith", 2)
    nug_old.validity.t_s = date(2015, 1, 1)

    nug_new = make_test_nugget("Bob Johnson became CEO in January 2019.", "Acme Robotics", "chiefExecutiveOfficer", "Bob Johnson", 2)
    nug_new.validity.t_s = date(2019, 1, 1)

    val = infer_validity(nug_old, doc_time="2015-01-01", sibling_nuggets=[nug_new])
    assert val.t_e == date(2019, 1, 1)

