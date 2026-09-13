# Retrieval benchmark v1 protocol (declared before comparison)

24 answerable queries over 12 original synthetic lessons, each with two sections.
All configurations use all queries; there is no held-out test set. Queries include
exact terms, natural questions, ambiguity and two lexical-gap probes. Human-readable
source judgments use grade 2 for direct evidence and 1 for supporting context.
An exact anchor must occur in the returned chunk to count that source as evidence.
Unjudged sources receive zero credit. Judgments are authored, not independently
annotated; this is a small development benchmark, not a generalization claim.

Score the first occurrence of each relevant source containing its anchor at its
actual chunk rank. Duplicate sources receive no further credit and still consume
rank positions. Recall@3 is unique relevant sources covered / judged sources;
Hit@3, MRR@3 and nDCG@3 use the same evidence-qualified rankings. nDCG uses gain
2^grade-1 and discounts log2(rank+1), with an ideal list of judged sources.

Candidate depth 12 per branch, fused pool at most 12, top-k 3; RRF constant 60;
BM25 k1=1.5, b=0.75. Compare frozen lexical, frozen weighted hybrid, BM25, local
fake dense, BM25+dense RRF, and RRF+local reranker. Compare baseline 900/120
boundary-aware character chunks with paragraph chunks at the same 900 cap using
BM25 alone. No parameter search or query-specific patches.

Promotion: primary nDCG@3 must increase by at least 0.02 versus the incumbent,
with no decrease in Recall@3 or MRR@3 (tolerance 1e-9). A reranker additionally
needs at least 0.02 nDCG over its no-reranker pair, and median warm retrieval
latency <= 2x that pair + 1 ms. A chunking change must clear the same quality
gate against the matching fixed-chunk configuration. Dense comparisons use a
fixed token-hash test embedder, not a learned semantic model: they cannot promote
a production dense or hybrid setting. Only the lexical comparison may promote
BM25. If promoted, embeddings may still be explicitly enabled as an experimental
hybrid configuration; the existing weighted pipeline remains selectable.

Timings are measured wall-clock stage durations on deterministic workloads,
not deterministic values or production service estimates. Report median/p95 warm
retrieval and index construction separately; excludes network, extraction and
SQLite IO. Zero paid API calls. Preserve the pre-upgrade baseline artifact.
