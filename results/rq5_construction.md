# Table 5: Automated Construction Pipeline Quality

| Stage | Metric | Score | Paper Benchmark (TimeQA) | Description |
|---|---|:---:|:---:|---|
| **Stage 1: Extraction** | Precision | 0.920 | 0.860 | Fraction of extracted nuggets that are valid factual assertions |
| | Atomicity | 0.960 | 0.943 | Fraction of nuggets expressing exactly one atomic statement |
| **Stage 2: Normalization** | Alias Resolution Recall | 0.850 | 0.600 | Correct mapping of alias variations to canonical entities |
| | False Merge Rate ↓ | 0.000 | 0.016 | Distinct entities incorrectly conflated into one |
| **Stage 3: Validity Inference** | Start Time Accuracy | 0.940 | 0.868 | Correct identification of assertion start dates |
| | End Time Detection Recall | 0.875 | 0.667 | Ability to detect when facts terminate or supersede |
| | End Time Accuracy | 0.980 | 0.955 | Precision of assigned expiration timestamps |
| **Stage 4: Conflict Detection** | Deprecation Precision | 1.000 | 1.000 | Older conflicting single-valued claims accurately deprecated |
| | Contestation Appropriateness | 1.000 | 1.000 | Equal uncorroborated conflicting claims marked Contested |
| | Missed Conflict Rate ↓ | 0.000 | 0.000 | Overlapping incompatible functional assertions detected |