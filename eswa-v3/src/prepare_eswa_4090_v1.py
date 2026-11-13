"""Remote-only preparation; label-independent fresh sampling and exact legacy retrieval."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, collections, hashlib, json
from pathlib import Path
from eswa_stage2_common import ROOT, load, save, save_rows, read_jsonl, sha, utc
from prepare_eswa_stage2_v1 import norm, bm25

C=load(ROOT/'configs/eswa_4090_validation_v1.json');D=ROOT/C['data'];O=ROOT/C['output']
def key(s):return hashlib.sha256((str(C['seed'])+'|'+s).encode()).hexdigest()
def initial_freeze():
    O.mkdir(parents=True,exist_ok=True);D.mkdir(parents=True,exist_ok=True)
    paths=['configs/eswa_4090_validation_v1.json','research/eswa_4090_validation_protocol_v1.md']
    save(O/'rules_freeze.json',{'utc':utc(),'state':'rules_frozen_before_new_scoring','files_sha256':{p:sha(ROOT/p) for p in paths}})

def split():
    for p,h in load(O/'rules_freeze.json')['files_sha256'].items():assert sha(ROOT/p)==h
    assert not (O/'sampling.json').exists()
    sources={'dev':ROOT/'data/eswa_supplement_v2/qasper-dev-v0.3.json','test':ROOT/'data/eswa_gate_stage2_v1/qasper-test-v0.3.json'}
    pools={s:load(p) for s,p in sources.items()}
    q_to_p={q['question_id']:pid for pool in pools.values() for pid,p in pool.items() for q in p['qas']}
    ids=set();texts=set();papers=set();files=[]
    paths=set()
    for directory in ['results','data']:
        for p in (ROOT/directory).rglob('*.jsonl'):
            rel=p.relative_to(ROOT)
            if C['output'] in str(rel) or C['data'] in str(rel) or 'raw' in rel.parts:continue
            if directory=='data' or any(s in p.stem.lower() for s in ['input','quer','label','sample']):paths.add(p)
    for p in sorted(paths):
        count=0
        with p.open(encoding='utf-8-sig') as f:
            for line in f:
                if not line.strip():continue
                r=json.loads(line)
                if not isinstance(r,dict):continue
                qid=r.get('id',r.get('_id',r.get('question_id')));q=r.get('question')
                if qid is not None and (isinstance(q,str) or str(qid) in q_to_p):
                    qid=str(qid);ids.add(qid);count+=1
                    if isinstance(q,str):texts.add(norm(q))
                    if r.get('paper_id'):papers.add(str(r['paper_id']))
                    if qid in q_to_p:papers.add(q_to_p[qid])
        if count:files.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'rows':count})
    eligible={};excluded={}
    for split,pool in pools.items():
        eligible[split]=[];excluded[split]=collections.Counter()
        for pid,p in pool.items():
            if pid in papers:excluded[split]['historical_paper']+=1;continue
            docs=[{'title':s['section_name'] or p['title'],'text':t,'paper_id':pid,'paragraph_id':f'{si}:{pi}'} for si,s in enumerate(p['full_text']) for pi,t in enumerate(s['paragraphs']) if t.strip()]
            qs=[q for q in p['qas'] if q['question'].strip() and q['question_id'] not in ids and norm(q['question']) not in texts]
            if len(docs)<10:excluded[split]['fewer_than_10_paragraphs']+=1;continue
            if not qs:excluded[split]['no_fresh_question_text']+=1;continue
            q=min(qs,key=lambda q:key('question|'+q['question_id']))
            eligible[split].append((pid,docs,q,split))
        eligible[split].sort(key=lambda r:key('paper|'+r[0]))
    available=eligible['dev']+eligible['test'];assert len(available)>=33,{'available':len(available),'excluded':excluded}
    n=min(192,len(available)-32);chosen=available[:n+32];qr=[];ql=[];newtexts=set()
    # Duplicate generic texts in the new sample are disclosed, never silently tuned away.
    for i,(pid,docs,q,split) in enumerate(chosen):
        role='evaluation' if i<n else 'calibration';scores=bm25(q['question'],docs)
        ix=sorted(range(len(docs)),key=lambda j:(-scores[j],j))[:100]
        qr.append({'id':q['question_id'],'question':q['question'],'paper_id':pid,'dataset':'qasper','source_split':split,'role':role,'documents':[docs[j] for j in ix],'bm25_scores':[scores[j] for j in ix]})
        ql.append({'id':q['question_id'],'paper_id':pid,'qas':q,'role':role,'source_split':split})
        newtexts.add(norm(q['question']))
    hp=ROOT/'data/eswa_gate_stage2_v1/hotpot_dev_distractor_v1.json';h=load(hp)
    eh=sorted([r for r in h if r['_id'] not in ids and norm(r['question']) not in texts],key=lambda r:key('hotpot|'+r['_id']))
    assert len(eh)>=232;hl=eh[:232]
    hq=[{'id':r['_id'],'question':r['question']} for r in hl]
    save_rows(D/'hotpot_queries.jsonl',hq);save_rows(D/'hotpot_labels.jsonl',hl);save_rows(D/'qasper_labels.jsonl',ql);save_rows(O/'qasper_inputs.jsonl',qr)
    save(D/'roles.json',{r['_id']:('evaluation' if i<200 else 'calibration') for i,r in enumerate(hl)})
    assert not {r['id'] for r in qr+hq}&ids and not {r['paper_id'] for r in qr}&papers
    save(O/'sampling.json',{'utc':utc(),'historical_ids':len(ids),'historical_papers':len(papers),'historical_question_texts':len(texts),'history_files':files,
         'qasper_source_hashes':{s:sha(p) for s,p in sources.items()},'hotpot_source_sha256':sha(hp),
         'qasper_eligible':{s:len(x) for s,x in eligible.items()},'qasper_excluded':excluded,
         'selected_counts':dict(collections.Counter(r['role']+'/'+r['dataset']+'/'+r['source_split'] for r in qr))|{'evaluation/hotpot/dev':200,'calibration/hotpot/dev':32},
         'historical_id_overlap':0,'historical_qasper_paper_overlap':0,'historical_question_text_overlap':0,
         'within_new_qasper_duplicate_text_count':len(qr)-len(newtexts),'selection_uses_answers':False,
         'evaluation_ids':[r['id'] for r in qr if r['role']=='evaluation']+[r['_id'] for r in hl[:200]],
         'calibration_ids':[r['id'] for r in qr if r['role']=='calibration']+[r['_id'] for r in hl[200:]]})
    old=read_jsonl(ROOT/'results/eswa_gate_stage2_v1/inputs.jsonl');cross=[r for r in old if r['role']=='evaluation'];assert len(cross)==392
    save_rows(D/'cross_hardware_inputs.jsonl',cross)
    warm=[r for r in read_jsonl(ROOT/'results/eswa_supplement_v2/inputs.jsonl') if r['dataset']=='qasper'][:2]
    save_rows(D/'warmup_inputs.jsonl',warm)
    baseline=[]
    for model in ['qwen','bge']:
        for r in read_jsonl(ROOT/f'results/eswa_gate_stage2_v1/requests_{model}.jsonl'):
            if r['method']=='serial' and r['id'] in {x['id'] for x in cross}:baseline.append(r)
    assert len(baseline)==1568;save_rows(D/'baseline_3090ti_serial.jsonl',baseline)
    for split,source in pools.items():
        wanted={r['paper_id'] for r in qr if r['source_split']==split}
        save(D/f'qasper_{split}_selected.json',{pid:source[pid] for pid in wanted})
    print(json.dumps({'state':'sampled','qasper_evaluation':n,'counts':load(O/'sampling.json')['selected_counts'],'historical_ids':len(ids),'historical_papers':len(papers),'eligible':{s:len(x) for s,x in eligible.items()}},ensure_ascii=False),flush=True)

def candidates():
    import numpy as np,torch
    queries=read_jsonl(D/'hotpot_queries.jsonl');retrieved={n:read_jsonl(O/n/'retrieved.jsonl') for n in ['bge','e5']}
    for rr in retrieved.values():assert [{'id':r['id'],'question':r['question']} for r in rr]==queries
    required={d['corpus_row'] for rr in retrieved.values() for r in rr for d in r['documents'][:100]}
    index=ROOT/'indices/beir_hotpot_bge_small_v1';manifest=load(index/'manifest.json');assert manifest['complete'];vectors={}
    for shard in manifest['shards']:
        start=shard['first_global_index'];needed=required.intersection(range(start,start+shard['documents']))
        if not needed:continue
        path=index/shard['vectors'];assert sha(path)==shard['vector_sha256'];matrix=np.load(path)
        for i in needed:vectors[i]=matrix[i-start].copy()
    qvec=np.load(O/'bge/query_vectors.npy');out=[];roles=load(D/'roles.json');torch.set_num_threads(2)
    for i,q in enumerate(queries):
        docs={}
        for name in ['e5','bge']:
            for d in retrieved[name][i]['documents'][:100]:
                clean={k:d[k] for k in ['corpus_id','corpus_row','title','text']}
                assert d['corpus_id'] not in docs or docs[d['corpus_id']]==clean;docs[d['corpus_id']]=clean
        values=list(docs.values());v=(torch.from_numpy(np.stack([vectors[d['corpus_row']] for d in values]))@torch.from_numpy(qvec[i])).numpy().tolist()
        scores={d['corpus_id']:s for d,s in zip(values,v)};values.sort(key=lambda d:(-scores[d['corpus_id']],d['corpus_row']))
        out.append({**q,'dataset':'hotpot','source_split':'dev','role':roles[q['id']],'documents':values})
    assert not torch.cuda.is_initialized();out+=read_jsonl(O/'qasper_inputs.jsonl');out.sort(key=lambda r:(r['role'],r['dataset'],key('order|'+r['id'])))
    assert len({r['id'] for r in out})==len(out);assert sum(r['role']=='calibration' for r in out)==64
    save_rows(O/'inputs.jsonl',out);save(O/'candidate_manifest.json',{'utc':utc(),'inputs_sha256':sha(O/'inputs.jsonl'),'counts':dict(collections.Counter(r['role']+'/'+r['dataset'] for r in out)),'retrieval_corpus_documents':manifest['documents'],'candidate_construction_hardware':'source RTX3090Ti retrieval, CPU union, identical stage2 implementation'})
    print(json.dumps(load(O/'candidate_manifest.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['initial_freeze','split','candidates']);globals()[p.parse_args().phase]()
