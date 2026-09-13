"""Offline retrieval evaluation, separate from agent routing evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
from pathlib import Path
from time import perf_counter

from . import retrieval_baseline
from .materials import chunk_text, search_material_store
from .retrieval import RetrievalConfig, tokens, text_digest

CORPUS_PATH = Path(__file__).resolve().parent.parent / 'benchmarks/retrieval/corpus.json'


class HashTestEmbedder:
    """256-dimensional signed token hashing. Test fixture, NOT semantic embeddings.

    No query judgments, synonyms, training, network, or Python randomized hash.
    """
    model_id = 'test:sha256-token-hash:256:v1'

    def embed_query(self, text):
        vector = [0.] * 256
        for token in tokens(text):
            digest = hashlib.sha256(token.encode()).digest()
            vector[int.from_bytes(digest[:2], 'big') % 256] += 1 if digest[2] % 2 else -1
        if not any(vector):
            vector[0] = 1
        return vector

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


def metrics(items, judgments, k=3):
    """Evidence-qualified source gains at actual chunk ranks; duplicates earn zero."""
    if k < 1 or not judgments or any(j['grade'] <= 0 for j in judgments.values()):
        raise ValueError('metrics require positive k and positive relevance judgments')
    seen = set()
    gains = []
    rr = 0.
    for rank, item in enumerate(items[:k], 1):
        source = item['document_id']
        judgment = judgments.get(source)
        grade = 0
        if judgment and source not in seen and judgment['anchor'] in item['text']:
            grade = judgment['grade']
            seen.add(source)
            if not rr:
                rr = 1 / rank
        gains.append((2 ** grade - 1) / math.log2(rank + 1))
    ideal = sum((2 ** j['grade'] - 1) / math.log2(rank + 1)
                for rank, j in enumerate(sorted(judgments.values(), key=lambda j: -j['grade'])[:k], 1))
    return {f'recall@{k}': len(seen) / len(judgments), f'hit@{k}': float(bool(seen)),
            f'mrr@{k}': rr, f'ndcg@{k}': sum(gains) / ideal}


def quality_gate(candidate, incumbent):
    return (candidate['ndcg@3'] >= incumbent['ndcg@3'] + .02 - 1e-9
            and candidate['recall@3'] >= incumbent['recall@3'] - 1e-9
            and candidate['mrr@3'] >= incumbent['mrr@3'] - 1e-9)


def promotion(configurations):
    baseline = configurations['legacy']['metrics']
    bm25 = configurations['bm25']['metrics']
    chosen = 'bm25' if quality_gate(bm25, baseline) else 'legacy'
    chunking = 'fixed'
    if chosen == 'bm25' and quality_gate(configurations['bm25_paragraph']['metrics'], bm25):
        chunking = 'paragraph'
    hybrid, reranked = configurations['hybrid'], configurations['hybrid_rerank']
    rerank_gate = (quality_gate(reranked['metrics'], hybrid['metrics']) and
                   reranked['latency_ms']['median'] <= 2 * hybrid['latency_ms']['median'] + 1)
    return {'default_mode': chosen, 'default_chunking': chunking,
            'reranker_pair_gate': rerank_gate, 'default_rerank': False,
            'dense_promotion_eligible': False,
            'reason': 'Only lexical evidence is eligible; hash test vectors do not validate production semantics.'}


def build_store(corpus, strategy='fixed'):
    embedder = HashTestEmbedder()
    chunks = []
    for d in corpus['materials']:
        for i, text in enumerate(chunk_text(d['text'], strategy=strategy)):
            chunks.append({'document_id':d['id'], 'source_name':d['source'], 'chunk_index':i,
                           'text':text, 'embedding':embedder.embed_query(text),
                           'embedding_model':embedder.model_id, 'embedding_text_sha256':text_digest(text)})
    return {'book_id':'synthetic-benchmark', 'documents':corpus['materials'], 'chunks':chunks}


def validate_corpus(corpus):
    docs = {d['id']:d for d in corpus['materials']}
    if len(docs) != len(corpus['materials']) or len({q['id'] for q in corpus['queries']}) != len(corpus['queries']):
        raise ValueError('duplicate benchmark identifiers')
    for q in corpus['queries']:
        if not q['judgments']:
            raise ValueError('missing judgments')
        for source, judgment in q['judgments'].items():
            if not judgment['anchor'] or judgment['anchor'] not in docs[source]['text']:
                raise ValueError('judgment anchor is absent')


def run_benchmark(repeats=7):
    if repeats < 1:
        raise ValueError('repeats must be positive')
    corpus = json.loads(CORPUS_PATH.read_text())
    validate_corpus(corpus)
    embedder = HashTestEmbedder()
    configurations = {}
    specs = [('legacy', 'legacy', 'fixed', False, False),
             ('legacy_weighted', 'legacy', 'fixed', True, False),
             ('bm25', 'bm25', 'fixed', False, False),
             ('dense', 'dense', 'fixed', True, False),
             ('hybrid', 'hybrid', 'fixed', True, False),
             ('hybrid_rerank', 'hybrid', 'fixed', True, True),
             ('bm25_paragraph', 'bm25', 'paragraph', False, False)]
    for name, mode, strategy, vector, rerank in specs:
        t = perf_counter()
        store = build_store(corpus, strategy)
        index_ms = (perf_counter() - t) * 1000
        config = RetrievalConfig(mode=mode, rerank=rerank)
        per_query, times, stages = [], [], {}
        for q in corpus['queries']:
            kwargs = {'query_embedding': embedder.embed_query(q['query']), 'embedding_model':embedder.model_id} if vector else {}
            result = search_material_store(store, q['query'], 3, config=config, **kwargs)
            for _ in range(repeats):
                trace = {}
                t = perf_counter()
                measured = search_material_store(store, q['query'], 3, config=config, trace=trace, **kwargs)
                times.append((perf_counter() - t) * 1000)
                assert result == measured, 'nondeterministic retrieval output'
                for stage, ms in trace.get('stages', {}).items():
                    stages.setdefault(stage, []).append(ms)
            per_query.append({'id':q['id'], 'category':q['category'], 'metrics':metrics(result['items'], q['judgments']),
                              'items':[{k:item[k] for k in ('document_id','chunk_index','score')} for item in result['items']]})
        configurations[name] = {
            'mode':mode, 'chunking':strategy, 'chunks':len(store['chunks']), 'top_k':3,
            'candidate_depth':12 if mode != 'legacy' else 'all', 'pool_size':12 if mode != 'legacy' else 'all',
            'rerank':rerank, 'embedding_model':embedder.model_id if vector else None,
            'metrics':{metric:statistics.mean(q['metrics'][metric] for q in per_query) for metric in per_query[0]['metrics']},
            'latency_ms':{'median':statistics.median(times), 'p95':sorted(times)[math.ceil(.95*len(times))-1]},
            'stage_median_ms':{s:statistics.median(v) for s,v in stages.items()},
            'fixture_build_ms':index_ms, 'queries':per_query}
    return {'benchmark_version':1, 'corpus_sha256':hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest(),
            'query_count':len(corpus['queries']), 'material_count':len(corpus['materials']),
            'environment':{'python':platform.python_version(), 'platform':platform.platform()},
            'timing_repeats':repeats, 'paid_calls':0, 'configurations':configurations,
            'promotion':promotion(configurations)}


def render_report(report):
    lines = ['# Retrieval comparison v1', '',
             'Offline token-hash vectors are test fixtures, not learned semantic embeddings. All 24 queries included.', '',
             '| Configuration | Recall@3 | Hit@3 | MRR@3 | nDCG@3 | median ms | p95 ms |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, config in report['configurations'].items():
        m, t = config['metrics'], config['latency_ms']
        lines.append(f"| {name} | {m['recall@3']:.6f} | {m['hit@3']:.6f} | {m['mrr@3']:.6f} | {m['ndcg@3']:.6f} | {t['median']:.3f} | {t['p95']:.3f} |")
    lines.extend(['', 'Promotion decision:', '', '```json', json.dumps(report['promotion'], indent=2), '```', '',
                  'Timing excludes API, SQLite IO and extraction; indexes are reconstructed within measured search. Fixture build time and stage medians are in JSON. Wall-clock values vary.', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('evaluation_results/retrieval_v1.json'))
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    report = run_benchmark(args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    args.output.with_suffix('.md').write_text(render_report(report))
    print(render_report(report))


if __name__ == '__main__':
    main()
