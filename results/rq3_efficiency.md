# Table 2: Efficiency Comparison Across Retrieval Systems

| System | Median Input Length (Tokens) | Token Reduction vs Passage | Retrieval Latency P50 (ms) | P95 (ms) |
|---|:---:|:---:|:---:|:---:|
| **BM25-Passage** | 108 | 0.0% | 2.82ms | 4.07ms |
| **Dense-Passage** | 108 | 0.0% | 1.07ms | 1.41ms |
| **Hybrid-Passage** | 108 | 0.0% | 1.02ms | 1.57ms |
| **Hybrid + TimeFilter** | 37 | 65.7% | 0.99ms | 1.97ms |
| **Hybrid + RecencyRerank** | 108 | 0.0% | 1.29ms | 2.94ms |
| **Hybrid + LatestSnapshot** | 108 | 0.0% | 1.61ms | 2.13ms |
| **Proposition-RAG** | 99 | 8.3% | 1.01ms | 1.18ms |
| **Proposition + TimeFilter** | 31 | 71.3% | 1.37ms | 2.34ms |
| **NuggetIndex-Active** | 83 | 23.1% | 10.09ms | 13.35ms |
| **NuggetIndex-Full** | 95 | 12.0% | 9.86ms | 13.80ms |