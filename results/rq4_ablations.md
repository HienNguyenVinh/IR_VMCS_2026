# Table 3: Ablation Study on Time-Aware Question Answering

| Configuration | Nugget Recall@20 | Δ Recall | Governance Score | Δ Governance |
|---|:---:|:---:|:---:|:---:|
| **NuggetIndex-Active** | 0.692 | — | 0.692 | — |
| **− validity filtering** | 0.812 | +12.0% | 0.597 | -9.5% |
| **− lifecycle states** | 0.692 | — | 0.692 | — |
| **− contested handling** | 0.712 | +2.0% | 0.652 | -4.0% |
| **− normalization** | 0.672 | -2.0% | 0.692 | — |
| **− nugget granularity (passages)** | 0.742 | +5.0% | 0.609 | -8.3% |
| **− all governance** | 0.832 | +14.0% | 0.561 | -13.1% |
# Table 4: Retrieval Mode Ablation (Hybrid vs Lexical-Only vs Semantic-Only)

| Mode | Recall@10 | Temporal Correctness | P50 Latency (ms) | Deployment Feasibility |
|---|:---:|:---:|:---:|---|
| **Hybrid (Default)** | .885 | 1.000 | 8.85ms | Standard server / GPU / Vector DB |
| **Lexical-Only** | .842 | 1.000 | 9.10ms | Edge / Browser / Zero-embedding cost |
| **Semantic-Only** | .790 | 1.000 | 9.86ms | Heavy embedding inference required |