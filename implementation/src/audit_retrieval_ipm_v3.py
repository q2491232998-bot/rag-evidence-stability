import json
import time

import numpy as np
from threadpoolctl import threadpool_limits

from common import ROOT, atomic_json, read_jsonl
from cqc_budget12_core import file_sha
from ipm_crossset_v3_core import check_prereg


def main():
    cfg,base,prereg=check_prereg();started=time.perf_counter();details={};canonical={}
    queries=read_jsonl(ROOT/cfg['data']/'queries.jsonl');inputs=read_jsonl(base/'inputs.jsonl')
    assert [{'id':r['id'],'question':r['question']} for r in inputs]==queries
    for name,index_name in cfg['indices'].items():
        folder=ROOT/index_name;m=json.loads((folder/'manifest.json').read_text());runtime=json.loads((base/name/'runtime.json').read_text())
        assert file_sha(folder/'manifest.json')==prereg['external_asset_sha256'][index_name+'/manifest.json']==runtime['index_manifest_sha256']
        assert m['complete'] and m['documents']==runtime['documents_searched']==cfg['corpus_documents'] and runtime['top_k']==200
        rows=read_jsonl(base/name/'retrieved.jsonl');q=np.load(base/name/'query_vectors.npy')
        assert [{'id':r['id'],'question':r['question']} for r in rows]==queries and len(q)==len(rows)
        wanted={d['corpus_row'] for r in rows for d in r['documents']}
        if name=='bge':wanted|={d['corpus_row'] for r in inputs for d in r['documents']}
        vectors={};best=[([],[]) for _ in range(4)];next_start=0
        for shard in m['shards']:
            path=folder/shard['vectors'];assert file_sha(path)==shard['vector_sha256']
            v=np.load(path);start=shard['first_global_index'];assert start==next_start
            next_start+=len(v);assert len(v)==shard['documents'] and v.dtype==np.float32 and np.isfinite(v).all()
            for i in wanted.intersection(range(start,start+len(v))):vectors[i]=v[i-start].copy()
            local=q[:4]@v.T
            for i,scores in enumerate(local):
                indices=np.arange(start,start+len(v));order=np.lexsort((indices,-scores))[:200]
                vals=np.concatenate((np.asarray(best[i][0],dtype=np.float32),scores[order]));inds=np.concatenate((np.asarray(best[i][1],dtype=np.int64),indices[order]))
                take=np.lexsort((inds,-vals))[:200];best[i]=(vals[take],inds[take])
        assert next_start==m['documents'] and set(vectors)==wanted
        maximum=0.;checks=[]
        for i,row in enumerate(rows):
            docs=row['documents'];assert len(docs)==len({d['corpus_row'] for d in docs})==200
            assert docs==sorted(docs,key=lambda d:(-d['score'],d['corpus_row']))
            actual=np.array([d['score'] for d in docs]);recalc=np.stack([vectors[d['corpus_row']] for d in docs])@q[i]
            error=float(np.max(abs(recalc-actual)));maximum=max(maximum,error);assert error<=cfg['retrieval_cpu_atol']
            if i<4:
                chosen={d['corpus_row'] for d in docs};vals,inds=best[i]
                excluded=[float(v) for v,j in zip(vals,inds) if int(j) not in chosen]
                gap=max(excluded)-float(min(recalc)) if excluded else 0.
                assert gap<=cfg['retrieval_cpu_atol']
                checks.append({'id':row['id'],'cpu_full_search_top200_set_equal':set(map(int,inds))==chosen,'ordered_equal':list(map(int,inds))==[d['corpus_row'] for d in docs],'excluded_advantage':gap,'score_max_error':error})
        if name=='bge':
            for i,row in enumerate(inputs):canonical[row['id']]=(np.stack([vectors[d['corpus_row']] for d in row['documents']])@q[i]).tolist()
        details[name]={'all_retrieved_score_checks':len(rows)*200,'max_error':maximum,'full_corpus_cpu_queries':checks,'documents':next_start}
    scores=read_jsonl(base/'canonical_scores.jsonl')
    original={name:read_jsonl(base/name/'retrieved.jsonl') for name in cfg['indices']}
    for i,(row,score) in enumerate(zip(inputs,scores)):
        assert row['id']==score['id'] and len(row['documents'])==len(score['original_bge_scores'])
        assert np.allclose(canonical[row['id']],score['original_bge_scores'],atol=cfg['retrieval_cpu_atol'],rtol=0)
        assert list(range(len(row['documents'])))==sorted(range(len(row['documents'])),key=lambda j:(-score['original_bge_scores'][j],row['documents'][j]['corpus_row']))
        expected={}
        for name in original:
            for d in original[name][i]['documents'][:100]:
                clean={k:d[k] for k in ['corpus_id','corpus_row','title','text']}
                assert d['corpus_id'] not in expected or expected[d['corpus_id']]==clean
                expected[d['corpus_id']]=clean
        assert expected=={d['corpus_id']:d for d in row['documents']}
    report={'state':'passed','preregistration_sha256':file_sha(base/'preregistration.json'),'seconds':time.perf_counter()-started,'retrievers':details,'labels_read':False,'all96_full_corpus_cpu_search':False,'independent_query_encoder_rerun':False,'canonical_union_reconstructed':True,'cpu_tolerance':cfg['retrieval_cpu_atol'],'cuda_used':False}
    atomic_json(base/'retrieval_audit.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
