"""NuggetIndex: Governed Atomic Retrieval for Maintainable RAG (SIGIR '26).

Reproduction and reference implementation.
"""

from .models import (
    Nugget,
    Fact,
    Validity,
    Epistemic,
    Provenance,
    Evidence,
    Kind,
    Status,
    Rank,
)
from .pipeline import NuggetIndex

__all__ = [
    "Nugget",
    "Fact",
    "Validity",
    "Epistemic",
    "Provenance",
    "Evidence",
    "Kind",
    "Status",
    "Rank",
    "NuggetIndex",
]

