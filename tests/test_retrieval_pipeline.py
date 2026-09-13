import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app import materials, retrieval, retrieval_baseline
from app.retrieval import BM25Index, ExactVectorIndex, RetrievalConfig, reciprocal_rank_fusion
from app.retrieval_evaluation import (HashTestEmbedder, build_store, metrics, promotion,
                                      quality_gate, run_benchmark, CORPUS_PATH)
from app.storage import get_storage


def chunk(text, name='a', vector=None):
    item = {'document_id':name, 'source_name':name+'.txt', 'chunk_index':0, 'text':text}
    if vector is not None:
        item.update(embedding=vector, embedding_model='fake')
    return item


def store(*chunks):
    return {'documents':[{'document_id':c['document_id']} for c in chunks], 'chunks':list(chunks)}


def test_bm25_frequency_saturation_and_length_normalization():
    chunks = [chunk('alpha beta'), chunk('alpha alpha', 'b'), chunk('alpha ' + 'beta '*30,'c')]
    scores = dict(BM25Index(chunks).search('alpha', 8))
    assert scores[1] > scores[0] > scores[2]
    no_norm = dict(BM25Index(chunks, b=0).search('alpha', 8))
    assert no_norm[0] == no_norm[2]
    assert no_norm[1] < 2 * no_norm[0]
    assert BM25Index(chunks).search('unknown', 8) == []


def test_bm25_formula_and_ties():
    index = BM25Index([chunk('alpha'), chunk('beta','b')])
    assert index.search('alpha', 8)[0][1] == pytest.approx(math.log(2))
    tied = [chunk('alpha','z'), chunk('alpha','a')]
    assert BM25Index(tied).search('alpha', 8)[0][0] == 1
    assert BM25Index(tied).search('alpha alpha', 8) == BM25Index(tied).search('alpha', 8)


def test_vector_mapping_model_filter_and_cosine():
    chunks = [chunk('first','a',[0,1]),chunk('second','b',[2,0]),chunk('missing','c')]
    index = ExactVectorIndex(chunks,'fake')
    assert index.search([1,0],8) == [(1,1)]
    assert ExactVectorIndex(chunks,'other').search([1,0],8) == []
    assert index.search([-1,0],8) == []
    with pytest.raises(ValueError):
        index.search([1,0,0],8)


@pytest.mark.parametrize('vector', [[], [0,0], [float('nan'),1], [float('inf'),1]])
def test_invalid_vectors_fail_closed(vector):
    with pytest.raises(ValueError):
        ExactVectorIndex([chunk('alpha',vector=vector)], 'fake')


def test_rrf_rank_fusion_deduplicates_and_ignores_score_scale():
    result = reciprocal_rank_fusion([[(0,1000),(1,.1)],[(1,.9),(2,.2)]])
    assert result[1] == pytest.approx(1/62 + 1/61)
    assert result[1] > result[0] > result[2]
    assert reciprocal_rank_fusion([[(0,1),(0,2)]]) == {0:1/61}


def test_hybrid_rerank_bounded_order_metadata_and_trace():
    data = store(*[chunk('alpha beta',str(i),[1,0]) for i in range(20)])
    class Reverse:
        def score(self, query, texts):
            assert len(texts) == 12
            return list(range(len(texts)))
    trace = {}
    result = materials.search_material_store(data,'alpha',3,[1,0],'fake',
        config=RetrievalConfig(mode='hybrid',rerank=True),reranker=Reverse(),trace=trace)
    assert len(result['items']) == 3
    assert result['items'][0]['document_id'] == trace['candidates'][0]['document_id']
    assert trace['candidates'][0]['fused_rank'] == 12
    assert trace['candidates'][0]['final_rank'] == 1
    assert trace['candidates'][0]['lexical_rank'] == 12
    assert trace['reranker_used']
    assert all('text' not in c for c in trace['candidates'])
    assert 'stages' not in result and 'query' not in result


@pytest.mark.parametrize('bad', ['exception','length','nan'])
def test_reranker_failure_preserves_fused_order(bad):
    class Broken:
        def score(self, query, texts):
            if bad == 'exception':
                raise RuntimeError('private provider message')
            return [] if bad == 'length' else [float('nan')]*len(texts)
    data = store(chunk('alpha beta'),chunk('alpha','b'))
    baseline = materials.search_material_store(data,'alpha',config=RetrievalConfig(mode='hybrid'))
    trace = {}
    result = materials.search_material_store(data,'alpha',config=RetrievalConfig(mode='hybrid',rerank=True),reranker=Broken(),trace=trace)
    assert result['items'] == baseline['items']
    assert not trace['reranker_used'] and any('reranker:' in e for e in trace['errors'])


def test_deterministic_local_reranker_phrase_behavior():
    scores = retrieval.CoverageReranker().score('passive voice', ['voice passive', 'passive voice', 'passive'])
    assert scores[1] > scores[0] > scores[2]


@pytest.mark.parametrize('mode', ['legacy','bm25','dense','hybrid'])
def test_modes_user_book_scope_replacement_deletion_and_rebuild(mode):
    db = get_storage()
    for user,book in [('alice','same'),('bob','same'),('alice','other')]:
        db.save_book_state({'book_id':book},user)
    embedder = HashTestEmbedder()
    doc = materials.add_material('same','a.md',b'alpha old\n\nbeta old',user_id='alice',embedding_provider=embedder)
    materials.add_material('same','b.md',b'foreign gamma',user_id='bob',embedding_provider=embedder)
    cfg = RetrievalConfig(mode=mode)
    def search(user,book,query):
        return materials.search_learning_materials_for_book(book,query,user_id=user,embedding_provider=embedder,config=cfg)['items']
    assert search('alice','same','alpha')
    assert not search('alice','other','alpha')
    assert not search('bob','same','alpha')
    replacement = materials.add_material('same','new.md',b'delta new\n\nepsilon new',user_id='alice',
        replace_document_id=doc['document_id'],chunking='paragraph',embedding_provider=embedder)
    assert replacement['document_id'] == doc['document_id']
    assert not search('alice','same','alpha')
    assert search('alice','same','delta')[0]['source_name'] == 'new.md'
    raw = db.load_material_store('same','alice')
    raw['chunks'][0].pop('embedding')
    db.save_material_store('same',raw,'alice')
    assert search('alice','same','delta')
    assert not materials.delete_material('same',doc['document_id'],user_id='bob')
    assert materials.delete_material('same',doc['document_id'],user_id='alice')
    assert not search('alice','same','delta')
    assert search('bob','same','foreign')


def test_cache_text_model_invalidation_and_no_query_reembedding():
    class Counting(HashTestEmbedder):
        calls = 0
        def embed_documents(self,texts):
            self.calls += 1
            return super().embed_documents(texts)
    db=get_storage(); db.save_book_state({'book_id':'book'})
    provider=Counting()
    materials.add_material('book','a.txt',b'alpha beta',embedding_provider=provider)
    for _ in range(2):
        materials.search_learning_materials_for_book('book','alpha',embedding_provider=provider)
    assert provider.calls == 1
    raw=db.load_material_store('book'); raw['chunks'][0]['text']='gamma delta'
    db.save_material_store('book',raw)
    materials.search_learning_materials_for_book('book','gamma',embedding_provider=provider)
    assert provider.calls == 2
    provider.model_id='test:new-model'
    materials.search_learning_materials_for_book('book','gamma',embedding_provider=provider)
    assert provider.calls == 3


def test_failed_embedding_batch_is_atomic():
    class Bad:
        model_id='fake'
        def embed_documents(self,texts):
            return [[1,0],[float('nan'),0]]
    chunks=[chunk('alpha'),chunk('beta','b')]; before=copy.deepcopy(chunks)
    with pytest.raises(ValueError):
        materials._cache_chunk_embeddings(chunks,Bad())
    assert chunks == before


def test_cache_fill_cannot_resurrect_concurrently_deleted_material():
    db=get_storage(); db.save_book_state({'book_id':'book'})
    doc=materials.add_material('book','a.txt',b'alpha beta',embedding_provider=None)
    class Deleting(HashTestEmbedder):
        def embed_documents(self,texts):
            materials.delete_material('book',doc['document_id'])
            return super().embed_documents(texts)
    result=materials.search_learning_materials_for_book('book','alpha',embedding_provider=Deleting())
    assert result['items'] == []
    assert db.load_material_store('book')['chunks'] == []
    assert result['retrieval']['fallbacks']


def test_concurrent_upload_conflict_preserves_winner():
    db=get_storage(); db.save_book_state({'book_id':'book'})
    class Competing(HashTestEmbedder):
        def embed_documents(self,texts):
            materials.add_material('book','winner.txt',b'winner note',embedding_provider=None)
            return super().embed_documents(texts)
    with pytest.raises(ValueError,match='concurrently'):
        materials.add_material('book','loser.txt',b'loser note',embedding_provider=Competing())
    assert [d['source_name'] for d in db.load_material_store('book')['documents']] == ['winner.txt']


@pytest.mark.parametrize('mode', ['legacy','bm25','dense','hybrid'])
def test_provider_unavailable_and_bad_query_fallback(mode):
    db=get_storage();db.save_book_state({'book_id':'book'})
    materials.add_material('book','a.txt',b'alpha beta',embedding_provider=None)
    class Broken(HashTestEmbedder):
        def embed_query(self,text):
            raise RuntimeError('unavailable')
    result=materials.search_learning_materials_for_book('book','alpha',embedding_provider=Broken(),config=RetrievalConfig(mode=mode))
    assert result['items'][0]['source_name'] == 'a.txt'
    if mode != 'bm25':
        assert result['retrieval']['fallbacks']


def test_disabled_provider_wins_over_cached_configuration(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','unused')
    monkeypatch.setenv('LANGBUDDY_EMBEDDINGS','disabled')
    assert materials.get_embedding_provider() is None


@pytest.mark.parametrize('filename,raw', [('notes.exe',b'text'),('notes.txt',b'  '),('notes.pdf',b'bad pdf')])
def test_invalid_ingestion_does_not_mutate_store(filename,raw):
    db=get_storage();db.save_book_state({'book_id':'book'})
    with pytest.raises(ValueError):
        materials.add_material('book',filename,raw,embedding_provider=None)
    assert db.load_material_store('book') is None


def test_chunking_preserves_baseline_and_bounds():
    text=('Paragraph alpha beta. '*80)+'\n\n# Next section\n'+('Gamma delta. '*90)
    assert materials.chunk_text(text) == retrieval_baseline.chunk_text(text)
    paragraphs=materials.chunk_text(text,strategy='paragraph')
    assert all(len(c)<=900 for c in paragraphs)
    assert not any('alpha' in c and 'Gamma' in c for c in paragraphs)
    with pytest.raises(ValueError):
        materials.chunk_text('abc',target_size=2,overlap=2)


def test_explicit_file_store_user_scope(tmp_path):
    materials.add_material('book','a.txt',b'alpha',materials_dir=tmp_path,user_id='alice',embedding_provider=None)
    assert materials.load_material_store('book',tmp_path,'bob')['chunks'] == []
    assert materials.load_material_store('book',tmp_path,'alice')['chunks']


def test_metrics_known_values_duplicates_and_evidence_required():
    judgments={'a':{'grade':2,'anchor':'yes'},'b':{'grade':1,'anchor':'yes'}}
    items=[chunk('no','a'),chunk('yes','a'),chunk('yes','a')]
    m=metrics(items,judgments)
    assert m['recall@3'] == .5 and m['mrr@3'] == .5 and m['hit@3'] == 1
    assert m['ndcg@3'] == pytest.approx((3/math.log2(3))/(3+1/math.log2(3)))
    assert metrics([],judgments)['ndcg@3'] == 0
    assert metrics([chunk('yes','a'),chunk('yes','b')],judgments)['ndcg@3'] == 1


def test_promotion_gate_rejects_ties_regressions_and_ineligible_dense():
    base={'ndcg@3':.5,'recall@3':.8,'mrr@3':.7}
    assert not quality_gate(base,base)
    assert quality_gate(base|{'ndcg@3':.52},base)
    assert not quality_gate(base|{'ndcg@3':.6,'recall@3':.79},base)
    configs={name:{'metrics':base.copy(),'latency_ms':{'median':1}} for name in
             ['legacy','bm25','bm25_paragraph','hybrid','hybrid_rerank']}
    configs['hybrid_rerank']['metrics']['ndcg@3']=.9
    assert promotion(configs)['default_mode'] == 'legacy'
    assert not promotion(configs)['dense_promotion_eligible']
    configs['bm25']['metrics']['ndcg@3']=.55
    assert promotion(configs)['default_mode'] == 'bm25'
    configs['hybrid_rerank']['latency_ms']['median']=10
    assert not promotion(configs)['reranker_pair_gate']


def test_benchmark_reproduces_frozen_baseline_and_committed_metrics():
    report=run_benchmark(repeats=1)
    prior=json.loads(Path('evaluation_results/retrieval_pre_upgrade.json').read_text())
    assert hashlib.sha256(Path('app/retrieval_baseline.py').read_bytes()).hexdigest()==prior['baseline_sha256']
    assert report['corpus_sha256']==prior['corpus_sha256']
    corpus=json.loads(CORPUS_PATH.read_text()); data=build_store(corpus)
    for q in corpus['queries']:
        assert retrieval_baseline.search_material_store(data,q['query'],3)==prior['results'][q['id']]
    committed=json.loads(Path('evaluation_results/retrieval_v1.json').read_text())
    for name,cfg in report['configurations'].items():
        assert cfg['metrics']==committed['configurations'][name]['metrics']
        assert len(cfg['queries'])==24
    assert report['promotion']==committed['promotion']
    assert retrieval.DEFAULT_CONFIG.mode==report['promotion']['default_mode']


def test_sm2_fixed_clock_regression():
    from app.sm2 import review_card
    now=datetime(2026,1,1,tzinfo=timezone.utc)
    card={}
    review_card(card,5,now); assert card['interval']==1 and card['reps']==1
    review_card(card,5,now); assert card['interval']==6 and card['reps']==2
    review_card(card,1,now); assert card['interval']==1 and card['reps']==0
    assert card['due_at']=='2026-01-02T00:00:00+00:00'


def test_missing_vector_snapshot_falls_back_to_bm25():
    trace={}
    result=materials.search_material_store(store(chunk('alpha')), 'alpha', query_embedding=[1,0],
        embedding_model='fake',config=RetrievalConfig(mode='dense'),trace=trace)
    assert result['items'][0]['match']=='lexical'
    assert trace['errors']


def test_upload_provider_initialization_failure_preserves_lexical(monkeypatch):
    db=get_storage();db.save_book_state({'book_id':'book'})
    def fail():
        raise RuntimeError('bad credentials')
    monkeypatch.setattr(materials,'get_embedding_provider',fail)
    doc=materials.add_material('book','a.txt',b'alpha beta')
    assert doc['embedding_status']=='unavailable'
    assert materials.search_learning_materials_for_book('book','alpha')['items']


def test_agent_tool_excludes_internals_and_bounds_text(monkeypatch):
    from types import SimpleNamespace
    from app import agent
    monkeypatch.setattr(agent,'search_learning_materials_for_book', lambda *a,**k:{
        'total_documents':1, 'retrieval':{'query':'private'},
        'items':[chunk('x'*1500)|{'score':1,'embedding':[1,0]}]})
    runtime=SimpleNamespace(context=SimpleNamespace(user_id='alice',book_id='book'))
    result=agent.search_learning_materials.func(runtime,query='alpha')
    assert set(result)=={'total_documents','items'}
    assert len(result['items'][0]['text'])==900
    assert set(result['items'][0])=={'document_id','source_name','chunk_index','text'}


def test_source_mapping_stable_on_order_change():
    chunks=[chunk('alpha','z',[1,0]),chunk('alpha','a',[1,0])]
    cfg=RetrievalConfig(mode='hybrid')
    first=materials.search_material_store(store(*chunks),'alpha',query_embedding=[1,0],embedding_model='fake',config=cfg)
    second=materials.search_material_store(store(*reversed(chunks)),'alpha',query_embedding=[1,0],embedding_model='fake',config=cfg)
    assert first==second
