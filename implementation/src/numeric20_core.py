import datetime
import json
import os

from common import ROOT, atomic_json, digest, read_jsonl
from batching19_core import check19, groups
from cqc_budget12_core import file_sha, source_path, write_new


def subset(rows,cfg):
    assert len(rows)==len({r['id'] for r in rows})==cfg['source_questions']
    return sorted(rows,key=lambda r:digest({'stage':cfg['stage'],'seed':cfg['seed'],'id':r['id']}))[:cfg['questions']]


def layouts(encodings,arm,pad,size=4):
    assert arm['geometry'] in ['serial','padded_single','batch4']
    lengths=list(map(len,encodings));plan=groups(lengths,size)
    if arm['geometry']=='serial':plan=[[i] for i in range(len(encodings))]
    result=[]
    for indices in plan:
        width=max(lengths[i] for i in indices)
        tensors={'input_ids':[[pad]*(width-lengths[i])+encodings[i] for i in indices],'attention_mask':[[0]*(width-lengths[i])+[1]*lengths[i] for i in indices]}
        if arm['explicit_positions']:tensors['position_ids']=[[0]*(width-lengths[i])+list(range(lengths[i])) for i in indices]
        if arm['geometry']=='padded_single':
            for offset,index in enumerate(indices):result.append({'indices':[index],'tensors':{k:[v[offset]] for k,v in tensors.items()}})
        else:result.append({'indices':indices,'tensors':tensors})
    return result


def schedule(rows,cfg):
    return [{'phase':phase,'id':row['id'],'arm':arm['name']} for arm in cfg['arms'] for phase,items in [('warmup',rows[:1]),('primary',rows)] for row in items]


def gate(cfg,engineering,bf16,fp32,contexts):
    informative=bf16>cfg['atol']
    ratio=fp32['mean_abs']/bf16 if bf16 else None
    attenuation=informative and ratio<=cfg['maximum_fp32_drift_ratio']
    maximum=fp32['max_abs']<=cfg['atol'];stable=contexts==cfg['required_fp32_context_changes']
    return {'engineering_passed':bool(engineering),'bf16_drift_informative':bool(informative),'fp32_to_bf16_mean_drift_ratio':ratio,'attenuation_passed':bool(attenuation),'fp32_max_residual_passed':bool(maximum),'fp32_context_identity_passed':bool(stable),'mechanism_screen_passed':bool(engineering and attenuation and maximum and stable),'reserved_expansion_authorized':False,'automatic_next_experiment':False}


def check20():
    cfg=json.loads((ROOT/'configs/numeric_stage20.json').read_text());base=ROOT/cfg['output'];m=json.loads((base/'manifest.json').read_text())
    for n,h in m['files_sha256'].items():assert file_sha(source_path(n))==h,n
    for n,h in m['inputs_sha256'].items():assert file_sha(base/n)==h,n
    return cfg,base,m


def prepare():
    _,_,prior=check19();cfg=json.loads((ROOT/'configs/numeric_stage20.json').read_text());base=ROOT/cfg['output'];temp=base.with_name(base.name+'.preparing')
    assert not base.exists() and not temp.exists()
    rows=subset(read_jsonl(ROOT/cfg['candidate_source']),cfg);assert len(rows)==8
    names=list(prior['files_sha256'])+['configs/numeric_stage20.json','research/stage20_numeric_protocol.md','research/stage20_sources.md','src/numeric20_core.py','src/run_numeric20.py','src/audit_numeric20.py','src/test_numeric20.py','src/queue_numeric20.py','src/batching19_core.py','src/audit_batching19.py','src/run_batching19.py',cfg['candidate_source'],cfg['prior_requests'],'results/batching_stage19_pilot32/decision.json']
    temp.mkdir();write_new(temp/'inputs.jsonl',rows);write_new(temp/'schedule.jsonl',schedule(rows,cfg))
    documents=sum(len(r['documents']) for r in rows)
    record={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files_sha256':{n:file_sha(source_path(n)) for n in sorted(set(names))},'inputs_sha256':{p.name:file_sha(p) for p in temp.iterdir()},'questions':8,'documents':documents,'ids':[r['id'] for r in rows],'conditions':9,'primary_scored_documents':documents*9,'warmup_scored_documents':len(rows[0]['documents'])*9,'primary_question_conditions':72,'warmup_question_conditions':9,'qa_generation_calls':0,'generation_started':False,'answer_labels_parsed':False,'new_diagnostic_outputs_seen':False,'all_questions_previously_used':True}
    atomic_json(temp/'manifest.json',record);os.rename(temp,base);check20()
    print(json.dumps({'state':'frozen','manifest_sha256':file_sha(base/'manifest.json'),'documents':documents,'primary_scored_documents':record['primary_scored_documents'],'warmup_scored_documents':record['warmup_scored_documents'],'ids':record['ids']}))


if __name__=='__main__':
    prepare()
