"""Unit tests for Algorithm 2: Conflict Detection and State Assignment."""

from datetime import date
from nuggetindex.conflict import ConflictDetector, jaccard_value
from nuggetindex.models import Epistemic, Evidence, Fact, Kind, Nugget, Provenance, Rank, Status, Validity
from nuggetindex.schema import Schema


def make_nugget(subject: str, predicate: str, obj: str, t_s: str, t_e: str = None, sources: int = 1) -> Nugget:
    evs = [
        Evidence(doc_id=f"doc_{i}", source_id=f"src_{i}", start=0, end=10, text=f"{subject} {predicate} {obj}")
        for i in range(sources)
    ]
    val = Validity(t_s=date.fromisoformat(t_s), t_e=date.fromisoformat(t_e) if t_e else None)
    return Nugget(
        kind=Kind.SEMANTIC_FACT,
        fact=Fact(subject, predicate, obj),
        validity=val,
        epistemic=Epistemic(),
        provenance=Provenance(evidence=evs),
        text=f"{subject} {predicate} {obj}",
        normalized_subject=subject.lower(),
    )


def test_jaccard_value_similarity():
    n1 = make_nugget("Acme", "chiefExecutiveOfficer", "Alice Smith", "2015-01-01")
    n2 = make_nugget("Acme", "chiefExecutiveOfficer", "Alice Smith", "2015-01-01")
    assert jaccard_value(n1, n2) == 1.0

    n3 = make_nugget("Acme", "chiefExecutiveOfficer", "Bob Johnson", "2019-01-01")
    assert jaccard_value(n1, n3) < 0.3


def test_deduplication_merge():
    detector = ConflictDetector(Schema())
    index = {}

    n1 = make_nugget("Acme", "chiefExecutiveOfficer", "Alice Smith", "2015-01-01", sources=1)
    detector.integrate(n1, index)
    assert len(index[n1.key]) == 1
    assert n1.num_sources == 1

    # Exact duplicate from second source -> should merge evidence
    n2 = make_nugget("Acme", "chiefExecutiveOfficer", "Alice Smith", "2015-01-01", sources=1)
    n2.provenance.evidence[0].doc_id = "doc_other"
    n2.provenance.evidence[0].source_id = "src_other"
    detector.integrate(n2, index)

    assert len(index[n1.key]) == 1
    assert index[n1.key][0].num_sources == 2
    assert index[n1.key][0].epistemic.rank == Rank.PREFERRED


def test_succession_non_overlapping():
    detector = ConflictDetector(Schema())
    index = {}

    # Alice: 2015 to 2019
    n1 = make_nugget("Acme", "chiefExecutiveOfficer", "Alice Smith", "2015-01-01", "2019-01-01", sources=2)
    detector.integrate(n1, index)

    # Bob: 2019 onwards (no overlap)
    n2 = make_nugget("Acme", "chiefExecutiveOfficer", "Bob Johnson", "2019-01-01", None, sources=2)
    detector.integrate(n2, index)

    assert len(index[n1.key]) == 2
    assert n1.status == Status.ACTIVE
    assert n2.status == Status.ACTIVE


def test_functional_conflict_succession_and_deprecation():
    detector = ConflictDetector(Schema())
    index = {}

    # Alice: 2015 onwards (open-ended), 2 sources
    n1 = make_nugget("Acme", "chiefExecutiveOfficer", "Alice Smith", "2015-01-01", None, sources=2)
    detector.integrate(n1, index)

    # Bob: 2019 onwards, 2 sources -> succession tightens Alice's interval to 2019
    n2 = make_nugget("Acme", "chiefExecutiveOfficer", "Bob Johnson", "2019-01-01", None, sources=2)
    detector.integrate(n2, index)

    assert n1.validity.t_e == date(2019, 1, 1)
    assert n1.status == Status.ACTIVE  # Valid historical predecessor
    assert n2.status == Status.ACTIVE
    assert n2.provenance.parent_id == n1.id

    # Uncorroborated intruder claim in 2021 (Dan Pham, 1 source) contradicts Bob (2 sources) -> Deprecated
    n3 = make_nugget("Acme", "chiefExecutiveOfficer", "Dan Pham", "2021-08-01", None, sources=1)
    detector.integrate(n3, index)
    assert n3.status == Status.DEPRECATED


def test_contested_unresolved():
    detector = ConflictDetector(Schema())
    index = {}

    # Source 1 says Seattle (1 source)
    n1 = make_nugget("Zentech", "headquarters", "Seattle", "2020-01-01", None, sources=1)
    detector.integrate(n1, index)

    # Source 2 says Austin (1 source) -> overlapping and neither has >= 2 sources
    n2 = make_nugget("Zentech", "headquarters", "Austin", "2020-01-01", None, sources=1)
    detector.integrate(n2, index)

    assert n1.status == Status.CONTESTED
    assert n2.status == Status.CONTESTED


def test_multi_valued_relation():
    schema = Schema()
    schema.add_predicate("boardMember", cardinality="multi")
    detector = ConflictDetector(schema)
    index = {}

    n1 = make_nugget("Acme", "boardMember", "Elena Rostova", "2018-01-01", None, sources=1)
    detector.integrate(n1, index)

    n2 = make_nugget("Acme", "boardMember", "Marcus Vance", "2020-01-01", None, sources=1)
    detector.integrate(n2, index)

    assert n1.status == Status.ACTIVE
    assert n2.status == Status.ACTIVE
