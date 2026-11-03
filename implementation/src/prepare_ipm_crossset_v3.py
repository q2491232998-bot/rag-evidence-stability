import argparse
import json

import numpy as np

from common import ROOT, atomic_json, read_jsonl
from cqc_budget12_core import file_sha, write_new
from ipm_crossset_v3_core import check_prereg, preregister, split, freeze_inputs, export_labels


def candidates():
    import torch
    cfg,base,prereg=check_prereg()
    queries=read_jsonl(ROOT/cfg['data']/'queries.jsonl')
    sources={n:read_jsonl(base/n/'retrieved.jsonl') for n in cfg['indices']}
    for rows in sources.values():assert [{'id':r['id'],'question':r['question']} for r in rows]==queries
    required={d['corpus_row'] for rows in sources.values() for r in rows for d in r['documents'][:100]}
    index=ROOT/cfg['indices']['bge'];manifest=json.loads((index/'manifest.json').read_text())
    assert manifest['complete'] and manifest['documents']==cfg['corpus_documents']
    vectors={}
    for shard in manifest['shards']:
        path=index/shard['vectors'];assert file_sha(path)==shard['vector_sha256']
        matrix=np.load(path);start=shard['first_global_index']
        for i in required.intersection(range(start,start+len(matrix))):vectors[i]=matrix[i-start].copy()
    assert set(vectors)==required
    qvectors=np.load(base/'bge/query_vectors.npy');torch.set_num_threads(2)
    rows=[];evidence=[]
    for pos,q in enumerate(queries):
        found={}
        for name in ['e5','bge']:
            for d in sources[name][pos]['documents'][:100]:
                clean={k:d[k] for k in ['corpus_id','corpus_row','title','text']}
                assert clean['corpus_id'] not in found or found[clean['corpus_id']]==clean
                found[clean['corpus_id']]=clean
        docs=list(found.values());scores=(torch.from_numpy(np.stack([vectors[d['corpus_row']] for d in docs]))@torch.from_numpy(qvectors[pos])).numpy()
        lookup={d['corpus_id']:float(v) for d,v in zip(docs,scores)}
        docs.sort(key=lambda d:(-lookup[d['corpus_id']],d['corpus_row']))
        assert 100<=len(docs)<=200
        rows.append({**q,'documents':docs,'source_document_indices':list(range(len(docs)))})
        evidence.append({'id':q['id'],'original_bge_scores':[lookup[d['corpus_id']] for d in docs]})
    assert not torch.cuda.is_initialized()
    write_new(base/'inputs.jsonl',rows);write_new(base/'canonical_scores.jsonl',evidence)
    print(json.dumps({'state':'candidates','questions':len(rows),'documents':sum(len(r['documents']) for r in rows),'cuda_initialized':False}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preregister','split','candidates','freeze','labels']);a=p.parse_args()
    {'preregister':preregister,'split':split,'candidates':candidates,'freeze':freeze_inputs,'labels':export_labels}[a.stage]()
