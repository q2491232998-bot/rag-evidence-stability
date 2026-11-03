import datetime
import itertools
import json
import os

from common import ROOT, atomic_json, digest, read_jsonl
from cqc_budget12_core import file_sha, source_path, write_new


def schedule(rows,cfg):
    perms=list(itertools.permutations(cfg['arms']))
    result=[]
    for repeat in range(cfg['repetitions']):
        ordered=sorted(enumerate(rows),key=lambda t:digest({'stage':21,'seed':cfg['seed'],'repeat':repeat,'id':t[1]['id']}))
        for pos,(i,row) in enumerate(ordered):
            arms=perms[i%len(perms)]
            if repeat%2:arms=arms[::-1]
            for slot,arm in enumerate(arms):result.append({'repeat':repeat,'position':pos,'slot':slot,'id':row['id'],'arm':arm})
    return result


def decision(cfg,engineering,matched,old):
    identity=matched['context_changes']==0 and matched['output_token_changes']==0
    numerical=matched['score_drift']['max']<=cfg['maximum_fp32_geometry_residual']
    g1=engineering and identity and numerical
    speed=lambda c:c['latency_saving']['point']>=cfg['minimum_latency_saving'] and c['latency_saving']['ci95'][0]>0
    quality=all(old['paired'][m]['ci95'][0]>=cfg['minimum_'+m+'_ci_lower'] for m in ['f1','em'])
    g2=g1 and speed(matched)
    return {'engineering_passed':bool(engineering),'matched_precision_identity_passed':bool(identity),'matched_precision_numeric_passed':bool(numerical),'repair_screen_passed':bool(g1),'matched_precision_speed_passed':bool(speed(matched)),'repair_and_speed_passed':bool(g2),'original_reference_quality_passed':bool(quality),'original_reference_speed_passed':bool(speed(old)),'deployment_screen_passed':bool(g2 and quality and speed(old)),'reserved_expansion_authorized':False,'automatic_next_experiment':False}


def check21():
    cfg=json.loads((ROOT/'configs/precision_stage21.json').read_text());base=ROOT/cfg['output'];m=json.loads((base/'manifest.json').read_text())
    for n,h in m['files_sha256'].items():assert file_sha(source_path(n))==h,n
    for n,h in m['inputs_sha256'].items():assert file_sha(base/n)==h,n
    return cfg,base,m


def prepare():
    from numeric20_core import check20
    from cost18_core import check18
    _,_,p20=check20();_,_,p18=check18()
    cfg=json.loads((ROOT/'configs/precision_stage21.json').read_text());base=ROOT/cfg['output'];temp=base.with_name(base.name+'.preparing')
    assert not base.exists() and not temp.exists()
    rows=sum([read_jsonl(ROOT/n) for n in cfg['candidate_sources']],[])
    assert len(rows)==len({r['id'] for r in rows})==cfg['questions']==128
    count=sum(len(r['documents']) for r in rows);assert count==cfg['expected_documents']==22330
    names=set(p20['files_sha256'])|set(p18['files_sha256'])|set(cfg['candidate_sources'])|set(cfg['prior_requests'])
    names.update(['configs/precision_stage21.json','research/stage21_precision_protocol.md','src/precision21_core.py','src/run_precision21.py','src/audit_precision21.py','src/test_precision21.py','src/queue_precision21.py','results/numeric_stage20_probe8/manifest.json','results/numeric_stage20_probe8/decision.json','results/numeric_stage20_probe8/requests.jsonl'])
    temp.mkdir();write_new(temp/'inputs.jsonl',rows);write_new(temp/'schedule.jsonl',schedule(rows,cfg))
    m={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files_sha256':{n:file_sha(source_path(n)) for n in sorted(names)},'inputs_sha256':{p.name:file_sha(p) for p in temp.iterdir()},'questions':128,'documents':count,'primary_generation_calls':128*3*2,'primary_scored_documents':count*3*2,'warmup_generation_calls':cfg['warmup_questions']*3,'warmup_scored_documents':sum(len(r['documents']) for r in rows[:cfg['warmup_questions']])*3,'answer_labels_parsed':False,'new_fp32_qa_outputs_seen':False,'generation_started':False,'all_questions_previously_used':True}
    atomic_json(temp/'manifest.json',m);os.rename(temp,base);check21();print(json.dumps({'state':'frozen','manifest_sha256':file_sha(base/'manifest.json'),**{k:v for k,v in m.items() if k not in ['files_sha256','inputs_sha256']}}))


if __name__=='__main__':prepare()
