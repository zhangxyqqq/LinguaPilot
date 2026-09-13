# Retrieval comparison v1

Offline token-hash vectors are test fixtures, not learned semantic embeddings. All 24 queries included.

| Configuration | Recall@3 | Hit@3 | MRR@3 | nDCG@3 | median ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| legacy | 0.875000 | 0.916667 | 0.916667 | 0.902186 | 0.244 | 0.268 |
| legacy_weighted | 0.895833 | 0.916667 | 0.916667 | 0.907924 | 0.590 | 0.637 |
| bm25 | 0.875000 | 0.916667 | 0.916667 | 0.902186 | 0.268 | 0.301 |
| dense | 0.770833 | 0.833333 | 0.833333 | 0.811613 | 0.402 | 0.436 |
| hybrid | 0.895833 | 0.916667 | 0.916667 | 0.907924 | 0.666 | 0.716 |
| hybrid_rerank | 0.895833 | 0.916667 | 0.916667 | 0.907924 | 0.812 | 0.947 |
| bm25_paragraph | 0.875000 | 0.916667 | 0.916667 | 0.900684 | 0.298 | 0.341 |

Promotion decision:

```json
{
  "default_mode": "legacy",
  "default_chunking": "fixed",
  "reranker_pair_gate": false,
  "default_rerank": false,
  "dense_promotion_eligible": false,
  "reason": "Only lexical evidence is eligible; hash test vectors do not validate production semantics."
}
```

Timing excludes API, SQLite IO and extraction; indexes are reconstructed within measured search. Fixture build time and stage medians are in JSON. Wall-clock values vary.
