import datetime
import json
import os

from common import ROOT, atomic_json, read_jsonl
from cost17_core import check17, make_schedule
from cqc_budget12_core import file_sha, source_path, write_new


def groups(lengths, size):
    assert isinstance(size,int) and not isinstance(size,bool) and size>0
    assert lengths and all(isinstance(n,int) and n>0 for n in lengths)
    ordered=sorted(range(len(lengths)),key=lambda i:(lengths[i],i))
    return [ordered[j:j+size] for j in range(0,len(ordered),size)]


def padded_batch(sequences,pad):
    assert sequences and all(sequences)
    width=max(map(len,sequences))
    return {'input_ids':[[pad]*(width-len(s))+list(s) for s in sequences], 'attention_mask':[[0]*(width-len(s))+[1]*len(s) for s in sequences], 'position_ids':[[0]*(width-len(s))+list(range(len(s))) for s in sequences]}


def decision(cfg,engineering,context_changes,token_changes,paired,cost):
    identity=context_changes==cfg['required_context_changes'] and token_changes==cfg['required_output_token_changes']
    quality=all(paired[m]['ci95'][0]>=cfg['minimum_'+m+'_ci_lower'] for m in ['f1','em'])
    latency=cost['point']>=cfg['minimum_latency_saving'] and cost['ci95'][0]>0
    return {'engineering_passed':bool(engineering),'operational_identity_passed':bool(identity),'quality_screen_passed':bool(quality),'latency_screen_passed':bool(latency),'pilot_screen_passed':bool(engineering and identity and quality and latency),'reserved_expansion_authorized':False,'automatic_next_experiment':False}


def check19():
    cfg=json.loads((ROOT/'configs/batching_stage19.json').read_text())
    base=ROOT/cfg['output'];m=json.loads((base/'manifest.json').read_text())
    for name,sha in m['files_sha256'].items():assert file_sha(source_path(name))==sha,name
    for name,sha in m['inputs_sha256'].items():assert file_sha(base/name)==sha,name
    return cfg,base,m


def prepare():
    old,oldbase,oldmanifest=check17()
    cfg=json.loads((ROOT/'configs/batching_stage19.json').read_text());base=ROOT/cfg['output'];tmp=base.with_name(base.name+'.preparing')
    assert not base.exists() and not tmp.exists()
    rows=read_jsonl(ROOT/cfg['candidate_source']);assert len(rows)==32 and sum(len(r['documents']) for r in rows)==5631
    for key in ['reader','reranker','context_tokens','top_k','max_new_tokens','generator_max_input_tokens','reranker_max_length','reranker_repeat_atol','minimum_f1_ci_lower','minimum_em_ci_lower','minimum_latency_saving','bootstrap_replicates','seed']:
        assert cfg[key]==old[key],key
    names=list(oldmanifest['files_sha256'])+['configs/batching_stage19.json','research/stage19_batching_protocol.md','research/stage19_prior_art.md','src/batching19_core.py','src/run_batching19.py','src/audit_batching19.py','src/test_batching19.py','src/queue_batching19.py','results/cost_stage17_pilot32/inputs.jsonl','results/cost_stage17_pilot32/manifest.json','results/cost_stage17_pilot32/requests.jsonl','results/cost_stage18_complement96/decision.json']
    tmp.mkdir();write_new(tmp/'inputs.jsonl',rows);write_new(tmp/'schedule.jsonl',make_schedule(rows,cfg))
    record={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files_sha256':{n:file_sha(source_path(n)) for n in sorted(set(names))},'inputs_sha256':{p.name:file_sha(p) for p in tmp.iterdir()},'questions':32,'documents':5631,'primary_generation_calls':128,'primary_scored_documents':22524,'prior_outcomes_known':True,'new_batch4_outputs_seen':False,'generation_started':False,'answer_labels_parsed':False}
    atomic_json(tmp/'manifest.json',record);os.rename(tmp,base);check19()
    print(json.dumps({'state':'frozen','manifest_sha256':file_sha(base/'manifest.json'),'questions':32,'primary_generation_calls':128,'primary_scored_documents':22524}))


if __name__=='__main__':
    prepare()
