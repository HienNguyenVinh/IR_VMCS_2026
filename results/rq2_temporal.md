# Table 1: Temporal Correctness and Retrieval Performance Across Systems

| System Category | System | Nugget Recall@20 | Temporal Correctness | Conflict Rate ↓ | Governance Score (TC - CR) ↑ |
|---|---|:---:|:---:|:---:|:---:|
| Passage | **BM25-Passage** | 0.923 | 0.077 | 0.923 | -0.846 |
| Passage | **Dense-Passage** | 0.923 | 0.077 | 0.923 | -0.846 |
| Passage | **Hybrid-Passage** | 0.923 | 0.077 | 0.923 | -0.846 |
| Passage | **Hybrid + TimeFilter** | 0.385 | 0.538 | 0.077 | 0.462 |
| Passage | **Hybrid + RecencyRerank** | 0.923 | 0.077 | 0.923 | -0.846 |
| Passage | **Hybrid + LatestSnapshot** | 0.769 | 0.154 | 0.692 | -0.538 |
| Proposition | **Proposition-RAG** | 0.923 | 0.077 | 0.846 | -0.769 |
| Proposition | **Proposition + TimeFilter** | 0.385 | 0.538 | 0.077 | 0.462 |
| NuggetIndex | **NuggetIndex-Active** | 0.692 | 0.692 | 0.000 | 0.692 |
| NuggetIndex | **NuggetIndex-Full** | 0.923 | 1.000 | 0.231 | 0.769 |