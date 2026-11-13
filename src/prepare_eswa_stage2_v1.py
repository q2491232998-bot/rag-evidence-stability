"""CPU data acquisition/provenance and candidate preparation for the frozen stage2."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import collections
import json
from pathlib import Path
import re
import shutil
import tarfile
import urllib.request
import urllib.error
import math
from eswa_stage2_common import *


def fetch(url,path,expected=None):
    if path.exists():
        if expected:assert sha(path)==expected
        return
    request=urllib.request.Request(url,headers={'User-Agent':'research-reproducibility/1.0'})
    try:
        response=urllib.request.urlopen(request,timeout=30)
    except (urllib.error.URLError,TimeoutError):
        if not url.startswith('https://huggingface.co/'):raise
        mirror=url.replace('https://huggingface.co/','https://hf-mirror.com/',1)
        print(json.dumps({'network_fallback':mirror}),flush=True)
        response=urllib.request.urlopen(urllib.request.Request(mirror,headers={'User-Agent':'research-reproducibility/1.0'}),timeout=120)
    with response,path.with_suffix(path.suffix+'.partial').open('wb') as f:
        shutil.copyfileobj(response,f,1024*1024)
    temp=path.with_suffix(path.suffix+'.partial')
    if expected:assert sha(temp)==expected,(str(path),'hash')
    temp.rename(path)


def acquire():
    check();DATA.mkdir(parents=True,exist_ok=True);BASE.mkdir(parents=True,exist_ok=True)
    registry=[]
    for name,url in [('qasper-test-and-evaluator-v0.3.tgz',CFG['qasper_url']),('hotpot_dev_distractor_v1.json',CFG['hotpot_url'])]:
        p=DATA/name;extra={}
        try:fetch(url,p)
        except urllib.error.URLError as error:
            assert name=='hotpot_dev_distractor_v1.json'
            import pyarrow.parquet as pq
            metadata=DATA/'hotpot_hf_metadata.json'
            fetch('https://huggingface.co/api/datasets/hotpotqa/hotpot_qa',metadata)
            info=load(metadata);revision=info['sha'];sibs=[r['rfilename'] for r in info['siblings']]
            val=[n for n in sibs if n.startswith('distractor/') and 'validation' in n and n.endswith('.parquet')]
            full=[n for n in sibs if n.startswith('fullwiki/') and 'validation' in n and n.endswith('.parquet')]
            assert len(val)==len(full)==1,(val,full)
            originals=[];crosschecks=[]
            for subpath in val+full:
                dest=DATA/(subpath.replace('/','--'));fetch(f'https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/{revision}/{subpath}',dest)
                rr=pq.read_table(dest).to_pylist();assert len(rr)==7405
                (originals if subpath in val else crosschecks).extend(rr)
            def signature(r):return (r['question'],r['answer'],r['supporting_facts'])
            cross={r['id']:signature(r) for r in crosschecks}
            assert all(cross[r['id']]==signature(r) for r in originals)
            converted=[{'_id':r['id'],'question':r['question'],'answer':r['answer'],'type':r['type'],'level':r['level'],
                        'supporting_facts':[list(x) for x in zip(r['supporting_facts']['title'],r['supporting_facts']['sent_id'])],
                        'context':[list(x) for x in zip(r['context']['title'],r['context']['sentences'])]} for r in originals]
            save(p,converted)
            extra={'official_server_error':str(error),'fallback_repo':'hotpotqa/hotpot_qa','revision':revision,
                   'official_split':'validation/dev','rows':7405,'all_ids_questions_answers_support_facts_match_fullwiki':True,
                   'original_json_byte_identity_verified':False,'transport':'Hugging Face with hf-mirror fallback'}
        registry.append({'file':name,'url':url,'sha256':sha(p),'bytes':p.stat().st_size,**extra})
    with tarfile.open(DATA/'qasper-test-and-evaluator-v0.3.tgz') as tar:
        match=[p for p in tar.getmembers() if Path(p.name).name=='qasper-test-v0.3.json'];assert len(match)==1
        out=DATA/'qasper-test-v0.3.json'
        if not out.exists():
            with tar.extractfile(match[0]) as src,out.open('xb') as dst:shutil.copyfileobj(src,dst)
    info_url='https://huggingface.co/api/models/BAAI/bge-reranker-v2-m3?blobs=true'
    meta=DATA/'bge_model_metadata.json'
    fetch(info_url,meta);info=load(meta);revision=info['sha'];assert len(revision)==40
    folder=ROOT/'models/BAAI--bge-reranker-v2-m3';folder.mkdir(exist_ok=True)
    needed={'config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','sentencepiece.bpe.model','README.md'}
    available={f['rfilename']:f for f in info['siblings']}
    assert {'config.json','model.safetensors','tokenizer.json'}<=set(available)
    assets=[]
    for name in sorted(needed&set(available)):
        f=available[name];expected=f.get('lfs',{}).get('sha256');dest=folder/name
        url=f'https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/{revision}/{name}'
        fetch(url,dest,expected)
        assets.append({'name':name,'sha256':sha(dest),'bytes':dest.stat().st_size,'lfs_sha256':expected})
        print(json.dumps({'downloaded_asset':name,'bytes':dest.stat().st_size}),flush=True)
    save(DATA/'acquisition.json',{'utc':utc(),'datasets':registry,'bge_revision':revision,'bge_assets':assets})
    print(json.dumps({'state':'acquired','bge_revision':revision,'qasper_papers':len(load(out)),'hotpot_dev':len(load(DATA/'hotpot_dev_distractor_v1.json'))}),flush=True)


def norm(text):return ' '.join(re.findall(r'\w+',text.lower()))


def history():
    ids=set();questions=set();papers=set();files=[]
    candidates=set((ROOT/'results').rglob('inputs.jsonl'))|set((ROOT/'results').rglob('queries.jsonl'))|set((ROOT/'data').rglob('*.jsonl'))
    for p in sorted(candidates):
        rel=p.relative_to(ROOT)
        if str(rel).startswith(CFG['output']) or str(rel).startswith(CFG['data']) or 'raw' in rel.parts:continue
        n=0
        for r in read_jsonl(p):
            if not isinstance(r,dict):continue
            q=r.get('question');qid=r.get('id',r.get('_id',r.get('question_id')))
            if isinstance(q,str) and qid is not None:
                ids.add(str(qid));questions.add(norm(q));n+=1
                if r.get('paper_id'):papers.add(str(r['paper_id']))
        if n:files.append({'path':str(rel),'sha256':sha(p),'rows':n})
    return ids,questions,papers,files


def bm25(question,docs):
    counts=[collections.Counter(re.findall(r'\w+',d['text'].lower())) for d in docs]
    df=collections.Counter(w for c in counts for w in c);lengths=[sum(c.values()) for c in counts];avg=sum(lengths)/len(lengths)
    words=set(re.findall(r'\w+',question.lower()))
    return [sum(math.log(1+(len(docs)-df[w]+.5)/(df[w]+.5))*c[w]*2.5/(c[w]+1.5*(.25+.75*l/max(avg,1))) for w in words if c[w]) for c,l in zip(counts,lengths)]


def split():
    check();assert not (BASE/'sampling.json').exists()
    used_ids,used_texts,used_papers,files=history()
    old=read_jsonl(ROOT/'results/eswa_supplement_v2/inputs.jsonl')
    used_papers|={r['paper_id'] for r in old if r['dataset']=='qasper'}
    source=load(DATA/'qasper-test-v0.3.json');eligible=[]
    for pid,p in source.items():
        if pid in used_papers:continue
        docs=[{'title':s['section_name'] or p['title'],'text':t,'paper_id':pid,'paragraph_id':f'{si}:{pi}'}
              for si,s in enumerate(p['full_text']) for pi,t in enumerate(s['paragraphs']) if t.strip()]
        qs=[q for q in p['qas'] if q['question'].strip() and q['question_id'] not in used_ids]
        if len(docs)>=10 and qs:eligible.append((pid,docs,qs))
    eligible.sort(key=lambda x:key('paper|'+x[0]));n=CFG['bge_calibration_papers']+CFG['qasper_evaluation_papers'];assert len(eligible)>=n
    qrows=[];qlabels=[];text_overlap=0
    for i,(pid,docs,qs) in enumerate(eligible[:n]):
        q=min(qs,key=lambda q:key('question|'+q['question_id']));scores=bm25(q['question'],docs)
        indices=sorted(range(len(docs)),key=lambda j:(-scores[j],j))[:CFG['qasper_candidate_cap']]
        role='bge_calibration' if i<CFG['bge_calibration_papers'] else 'evaluation'
        qrows.append({'id':q['question_id'],'question':q['question'],'paper_id':pid,'dataset':'qasper','role':role,
                      'documents':[docs[j] for j in indices],'bm25_scores':[scores[j] for j in indices]})
        qlabels.append({'id':q['question_id'],'paper_id':pid,'qas':q,'role':role})
        text_overlap+=norm(q['question']) in used_texts
    dev=load(DATA/'hotpot_dev_distractor_v1.json')
    eligible_h=[r for r in dev if r['_id'] not in used_ids and norm(r['question']) not in used_texts]
    eligible_h.sort(key=lambda r:key('hotpot|'+r['_id']));chosen=eligible_h[:CFG['hotpot_dev_questions']];assert len(chosen)==200
    save_rows(DATA/'hotpot_queries.jsonl',[{'id':r['_id'],'question':r['question']} for r in chosen])
    save_rows(DATA/'hotpot_labels.jsonl',chosen);save_rows(DATA/'qasper_labels.jsonl',qlabels)
    save_rows(BASE/'qasper_inputs.jsonl',qrows)
    save(BASE/'sampling.json',{'utc':utc(),'historical_ids':len(used_ids),'historical_papers':len(used_papers),'history_files':files,
         'qasper_original_papers':len(source),'qasper_eligible_papers':len(eligible),'qasper_new_papers':n,
         'qasper_paper_overlap':0,'qasper_id_overlap':0,'qasper_generic_question_text_overlap':text_overlap,
         'calibration_ids':[r['id'] for r in qrows if r['role']=='bge_calibration'],
         'qasper_evaluation_ids':[r['id'] for r in qrows if r['role']=='evaluation'],
         'hotpot_evaluation_ids':[r['_id'] for r in chosen],'hotpot_id_text_overlap':0,
         'source_acquisition_sha256':sha(DATA/'acquisition.json'),'selection_uses_answers':False})
    print(json.dumps({'state':'split','qasper_papers':n,'qasper_evaluation':192,'bge_calibration':64,'hotpot_official_dev':200,'generic_qasper_text_overlap':text_overlap}),flush=True)


def candidates():
    check();import numpy as np
    queries=read_jsonl(DATA/'hotpot_queries.jsonl');retrieved={n:read_jsonl(BASE/n/'retrieved.jsonl') for n in ['bge','e5']}
    for rr in retrieved.values():assert [{'id':r['id'],'question':r['question']} for r in rr]==queries
    required={d['corpus_row'] for rr in retrieved.values() for r in rr for d in r['documents'][:100]}
    index=ROOT/'indices/beir_hotpot_bge_small_v1';manifest=load(index/'manifest.json');assert manifest['complete']
    vectors={}
    for shard in manifest['shards']:
        start=shard['first_global_index'];needed=required.intersection(range(start,start+shard['documents']))
        if not needed:continue
        path=index/shard['vectors'];assert sha(path)==shard['vector_sha256'];matrix=np.load(path)
        for i in needed:vectors[i]=matrix[i-start].copy()
    qvec=np.load(BASE/'bge/query_vectors.npy');out=[]
    import torch
    torch.set_num_threads(2)
    for i,q in enumerate(queries):
        docs={}
        for name in ['e5','bge']:
            for d in retrieved[name][i]['documents'][:100]:
                clean={k:d[k] for k in ['corpus_id','corpus_row','title','text']}
                assert d['corpus_id'] not in docs or docs[d['corpus_id']]==clean
                docs[d['corpus_id']]=clean
        values=list(docs.values());v=(torch.from_numpy(np.stack([vectors[d['corpus_row']] for d in values]))@torch.from_numpy(qvec[i])).numpy().tolist()
        scores={d['corpus_id']:s for d,s in zip(values,v)}
        values.sort(key=lambda d:(-scores[d['corpus_id']],d['corpus_row']))
        out.append({**q,'dataset':'hotpot','role':'evaluation','documents':values})
    assert not torch.cuda.is_initialized()
    out+=read_jsonl(BASE/'qasper_inputs.jsonl');out.sort(key=lambda r:(r['role'],r['dataset'],key('order|'+r['id'])))
    assert len(out)==456 and len({r['id'] for r in out})==456
    save_rows(BASE/'inputs.jsonl',out)
    save(BASE/'candidate_manifest.json',{'utc':utc(),'inputs_sha256':sha(BASE/'inputs.jsonl'),'questions':len(out),
         'evaluation_questions':392,'calibration_questions':64,'hotpot_candidates':sum(len(r['documents']) for r in out if r['dataset']=='hotpot'),
         'qasper_candidates':sum(len(r['documents']) for r in out if r['dataset']=='qasper'),'retrieval_corpus_documents':manifest['documents']})
    print(json.dumps({'state':'candidates_ready','questions':len(out)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['acquire','split','candidates']);args=p.parse_args();globals()[args.phase]()
