"""Freeze all scoring inputs/code/assets and validate the fixed gate against development fixtures."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
import tarfile
import argparse
from eswa_stage2_common import *


def main(gate_check_only=False):
    check();assert not (BASE/'freeze.json').exists()
    if not gate_check_only:
        for path,want in load(BASE/'definition_lock.json')['sha256'].items():assert sha(ROOT/path)==want,path
    from transformers import AutoTokenizer
    from eswa_gate_methods_v2 import order,flagged
    from run_eswa_stage2_v1 import Packer
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen--Qwen3-8B'),local_files_only=True)
    oldbase=ROOT/'results/eswa_gate_20261110_v1'
    cases=load(oldbase/'five_case_diagnosis/diagnosis.json')['cases']
    oldrows={r['id']:r for r in read_jsonl(ROOT/'results/eswa_supplement_v2/inputs.jsonl')}
    oldscores={(r['id'],r['arm']):r for r in read_jsonl(ROOT/'results/eswa_supplement_v2/scores_qwen.jsonl') if r['rep']==0}
    checks=[]
    for case in cases:
        qid=case['question_id'];a=oldscores[qid,'bf16_serial'];b=oldscores[qid,'bf16_batch4'];row=oldrows[qid]
        for version in ['v1','v2']:
            values=list(b['scores']);marked=set();packer=Packer(row['documents'],tok)
            while True:
                fresh,_=flagged(values,marked,packer,.25,version)
                if not fresh:break
                for i in fresh:values[i]=a['scores'][i]
                marked.update(fresh)
            result=packer(order(values))
            if version=='v1':assert result['selected']==case['v1_selected'] and sorted(marked)==case['v1_rescored']
            checks.append({'id':qid,'version':version,'serial_context_equal':result['context']==a['packs']['512']['context'],'rescored':sorted(marked)})
    # Simple strict-threshold and nonadjacent enumeration tests.
    class Toy:
        def __call__(self,ranking):return {'context':str([x for x in ranking[:10] if x!=1])}
    fresh,_=flagged([3.,2.,1.],set(),Toy(),1.,'v2');assert fresh==[]
    a=[3.,2.99,2.98];f1,_=flagged(a,set(),Toy(),.25,'v1');f2,_=flagged(a,set(),Toy(),.25,'v2')
    assert set(f1)<=set(f2)
    if gate_check_only:
        save(BASE/'development_gate_checks.json',{'utc':utc(),'stage1_still_failed':True,'checks':checks,'new_evaluation_scored':False})
        print(json.dumps({'development_checks':checks}),flush=True)
        return
    inputs=read_jsonl(BASE/'inputs.jsonl');sam=load(BASE/'sampling.json');cand=load(BASE/'candidate_manifest.json')
    assert sha(BASE/'inputs.jsonl')==cand['inputs_sha256'] and len(inputs)==456
    assert len([r for r in inputs if r['role']=='evaluation'])==392
    assert len({r['paper_id'] for r in inputs if r['dataset']=='qasper'})==256
    names=['configs/eswa_gate_stage2_v1.json','research/eswa_gate_stage2_authorization_20261110.md',
           'research/eswa_gate_stage2_protocol_20261110_v1.md','src/eswa_gate_methods_v2.py','src/eswa_stage2_common.py',
           'src/prepare_eswa_stage2_v1.py','src/run_eswa_stage2_v1.py','src/freeze_eswa_stage2_v1.py',
           'src/queue_eswa_stage2_v1.py','src/common.py','src/cqc_budget12_core.py','src/qwen_student.py','src/batching19_core.py',
           'configs/qwen512_v1.json','results/eswa_gate_stage2_v1/inputs.jsonl','results/eswa_gate_stage2_v1/sampling.json',
           'results/eswa_gate_stage2_v1/candidate_manifest.json','data/eswa_gate_stage2_v1/acquisition.json',
           'data/eswa_gate_stage2_v1/qasper_labels.jsonl','data/eswa_gate_stage2_v1/hotpot_labels.jsonl',
           'src/eswa_stage2_metrics.py','data/eswa_gate_stage2_v1/qasper_evaluator.py','src/run_idle_serial.py',
           'results/eswa_gate_stage2_v1/definition_lock.json','src/check_eswa_stage2_metrics.py']
    assets={}
    for record,repo in [(load(ROOT/'configs/cqc_budget_stage12_reranker_assets.json'),CFG['qwen']['repo']),
                        (load(ROOT/'logs/qwen3_reader_stage10_assets_verified.json'),CFG['reader']['repo'])]:
        for file in record['files']:
            name='models/'+repo.replace('/','--')+'/'+file['name'];assert sha(ROOT/name)==file['sha256'];assets[name]=file['sha256']
    acquisition=load(DATA/'acquisition.json')
    assert acquisition['bge_revision']==CFG['bge']['revision']
    for entry in acquisition['datasets']:assert sha(DATA/entry['file'])==entry['sha256']
    with tarfile.open(DATA/'qasper-test-and-evaluator-v0.3.tgz') as archive:
        evaluator=archive.extractfile('qasper_evaluator.py').read()
    destination=DATA/'qasper_evaluator.py'
    if destination.exists():assert destination.read_bytes()==evaluator
    else:
        with destination.open('xb') as f:f.write(evaluator)
    for file in acquisition['bge_assets']:
        name='models/BAAI--bge-reranker-v2-m3/'+file['name'];assert sha(ROOT/name)==file['sha256'];assets[name]=file['sha256']
    frozen={'utc':utc(),'files_sha256':{name:sha(ROOT/name) for name in names},'assets_sha256':assets,
            'qwen_tau':.25,'bge_revision':acquisition['bge_revision'],'gate_v2':'all unordered pairs in current top12; only adjacency rule differs',
            'development_fixture_checks':checks,'old_128_are_development':True,'stage1_status':'failed_86.49_percent',
            'continuation':'explicit_user_authorization_not_a_pass','new_evaluation_outputs_seen':False,'write_manuscript':False}
    save(BASE/'freeze.json',frozen)
    print(json.dumps({'state':'frozen','bge_revision':acquisition['bge_revision'],'fixtures':checks,'freeze_sha256':sha(BASE/'freeze.json')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gate-check-only',action='store_true');args=p.parse_args();main(args.gate_check_only)
