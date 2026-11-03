import datetime
import json
import os

from common import ROOT, atomic_json, digest, read_jsonl
from cqc_budget12_core import file_sha, source_path, write_new
from cost17_core import build_ranks, make_schedule, select_indices


def configuration():
    return json.loads((ROOT/'configs/cost_stage18.json').read_text())


def complement(source, excluded, expected=96):
    ids=[r['id'] for r in source]
    excluded_ids=[r['id'] for r in excluded]
    assert len(ids)==len(set(ids))==128
    assert len(excluded_ids)==len(set(excluded_ids))==32
    assert set(excluded_ids)<=set(ids)
    remaining=[r for r in source if r['id'] not in set(excluded_ids)]
    assert len(remaining)==expected
    return remaining


def project(source):
    indices=source['pools']['dual_dense_qwen']
    assert indices==sorted(set(indices))
    return {'id':source['id'],'question':source['question'],'documents':[{k:source['documents'][i][k] for k in ['corpus_id','corpus_row','title','text']} for i in indices],'source_document_indices':indices}


def score_reference(rows, originals, runtime):
    cached={}
    for name,sha in runtime['cache_sources'].items():
        assert file_sha(source_path(name))==sha
        for item in read_jsonl(source_path(name)):
            if item['key'] in cached:
                assert cached[item['key']]['score']==item['score']
            cached[item['key']]=item
    result=[]
    for row in rows:
        values=[]
        for doc in row['documents']:
            key=digest({'input':{'question':row['question'],'passages':[{'title':doc['title'],'text':doc['text']}]},'scorer':runtime['scorer_hash']})
            entry=cached[key]
            assert entry['question_id']==row['id'] and entry['corpus_id']==doc['corpus_id']
            values.append(entry['score'])
        order=sorted(range(len(values)),key=lambda i:(-values[i],i))
        assert [row['source_document_indices'][i] for i in order]==originals[row['id']]['rankings']['dual_dense_qwen']
        result.append({'id':row['id'],'scores':values,'source':'hash-verified Stage08/09 pointwise scorer caches; audit only, not runner input'})
    return result


def check18():
    cfg=configuration();base=ROOT/cfg['output']
    manifest=json.loads((base/'manifest.json').read_text())
    for name,sha in manifest['files_sha256'].items():
        assert file_sha(source_path(name))==sha,name
    for name,sha in manifest['inputs_sha256'].items():
        assert file_sha(base/name)==sha,name
    old=json.loads((ROOT/'configs/cost_stage17.json').read_text())
    for key in cfg['unchanged_keys']:
        assert cfg[key]==old[key],key
    return cfg,base,manifest


def prepare():
    cfg=configuration();base=ROOT/cfg['output'];staging=base.with_name(base.name+'.preparing')
    assert not base.exists() and not staging.exists()
    sources=read_jsonl(source_path(cfg['candidate_source']))
    prior=read_jsonl(ROOT/cfg['excluded_source'])
    remaining=complement(sources,prior)
    rows=[project(r) for r in remaining]
    assert all(project(r)==p for r in sources for p in prior if r['id']==p['id'])
    runtime=json.loads(source_path(cfg['score_runtime']).read_text())
    assert runtime['candidate_hash']==digest(sources)
    reference=score_reference(rows,{r['id']:r for r in sources},runtime)
    ranks=build_ranks(rows,cfg)
    files=['configs/cost_stage18.json','research/stage18_replication_protocol.md','src/cost18_core.py','src/run_cost18.py','src/audit_cost18.py','src/test_cost18.py','src/queue_cost18.py','src/cost17_core.py','src/run_cost17.py','src/audit_cost17.py','src/test_cost17.py','configs/cost_stage17.json','src/common.py','src/cqc_budget12_core.py','src/run_cqc_budget12.py','src/audit_cqc_budget12.py','src/qwen_student.py','src/utility_student.py','src/cqc_sampling12.py','src/run_idle_serial.py','configs/qwen512_v1.json','configs/cqc_budget_stage12_reranker_assets.json','logs/qwen3_reader_stage10_assets_verified.json',cfg['candidate_source'],cfg['excluded_source'],cfg['score_runtime'],cfg['bge_source'],cfg['e5_source'],'references/hotpot_eval.txt','data/stage08_confirmation/labels.jsonl','results/cost_stage17_pilot32/manifest.json','results/cost_stage17_pilot32/analysis_local/question_pairs.jsonl','results/cost_stage17_pilot32/decision.json']+list(runtime['cache_sources'])
    staging.mkdir()
    for name,items in [('inputs.jsonl',rows),('retrieval_ranks.jsonl',ranks),('schedule.jsonl',make_schedule(rows,cfg)),('audit_only_prior_scores.jsonl',reference)]:
        write_new(staging/name,items)
    count=sum(len(r['documents']) for r in rows)
    record={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files_sha256':{n:file_sha(source_path(n)) for n in files},'inputs_sha256':{p.name:file_sha(p) for p in staging.iterdir()},'generation_started':False,'answer_labels_parsed':False,'prior_original_scores_inspected_for_provenance':True,'prior_original_answers_exist':True,'new_rrf64_complement_answers_seen':False,'questions':len(rows),'documents':count,'excluded_ids':[r['id'] for r in prior],'primary_generation_calls':len(rows)*2*cfg['repetitions'],'primary_reranker_calls':cfg['repetitions']*(count+len(rows)*cfg['shortlist_size'])}
    atomic_json(staging/'manifest.json',record);os.rename(staging,base)
    check18()
    print(json.dumps({'state':'frozen','manifest_sha256':file_sha(base/'manifest.json'),'questions':len(rows),'documents':count,'primary_generation_calls':record['primary_generation_calls'],'primary_reranker_calls':record['primary_reranker_calls']}))


if __name__=='__main__':
    prepare()
