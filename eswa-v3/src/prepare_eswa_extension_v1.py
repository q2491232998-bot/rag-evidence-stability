"""CPU-only epsilon estimation and immutable experiment freeze before any new scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections
import random
from eswa_extension_common_v1 import *

def main():
    check();assert not OUT.exists(),'Never overwrite an existing supplement'
    from run_eswa_stage2_v1 import verify_freeze
    verify_freeze()
    assert load(STAGE2/'analysis/audit.json')['state']=='complete'
    old=ROOT/'results/eswa_supplement_v2';old_inputs=read_jsonl(old/'inputs.jsonl')
    qwen={(r['id'],r['arm']):r for r in read_jsonl(old/'scores_qwen.jsonl') if r['rep']==0 and r['arm'] in ['bf16_serial','bf16_batch4']}
    bge={(r['id'],r['method']):r for r in read_jsonl(STAGE2/'bge_calibration_requests.jsonl')}
    cal=[r for r in read_jsonl(STAGE2/'inputs.jsonl') if r['role']=='bge_calibration']
    assert len(old_inputs)==256 and len(cal)==64
    exports=[];epsilon={}
    for kind,rows,values,serial,batch in [('qwen',old_inputs,qwen,'bf16_serial','bf16_batch4'),('bge',cal,bge,'serial','batch4')]:
        for row in rows:
            a=values[row['id'],serial]['scores'];b=values[row['id'],batch]['scores']
            assert len(a)==len(b)==len(row['documents'])
            errors=[abs(x-y) for x,y in zip(a,b)];idx=max(range(len(errors)),key=lambda i:errors[i])
            exports.append(dict(model=kind,id=row['id'],dataset=row['dataset'],candidate_count=len(a),rep=0,batch_size=4,max_absolute_error=errors[idx],maximum_candidate_index=idx,serial_at_max=a[idx],batch_at_max=b[idx]))
        rr=[r for r in exports if r['model']==kind];maximum=max(r['max_absolute_error'] for r in rr)
        epsilon[kind]=dict(epsilon=1.5*maximum,development_maximum_error=maximum,questions=len(rr),candidates=sum(r['candidate_count'] for r in rr),source_batch=4,source_rep=0)
    # Independent finite synthetic verification including ties, zero epsilon and N<10.
    rng=random.Random(20261111);cases=0
    for n in [1,2,9,10,11,30,100]:
        for eps in [0,.125,.5,2.]:
            for _ in range(200):
                s=[rng.randint(-16,16)/4 for i in range(n)]
                b=[v+rng.choice([-eps,0,eps]) for v in s]
                chosen,kth=band_indices(b,eps)
                expected=sorted(range(n),key=lambda i:(-s[i],i))[:10]
                got=sorted(chosen,key=lambda i:(-s[i],i))[:10]
                assert got==expected;cases+=1
    names=['configs/eswa_extension_20261111_v1.json','research/eswa_extension_20261111_protocol_v1.md',
           'src/eswa_extension_common_v1.py','src/prepare_eswa_extension_v1.py','src/run_eswa_extension_v1.py','src/queue_eswa_extension_v1.py',
           'results/eswa_gate_stage2_v1/freeze.json','results/eswa_gate_stage2_v1/inputs.jsonl','results/eswa_gate_stage2_v1/requests_qwen.jsonl',
           'results/eswa_gate_stage2_v1/requests_bge.jsonl','results/eswa_gate_stage2_v1/bge_calibration_requests.jsonl',
           'results/eswa_gate_stage2_v1/answers.jsonl','results/eswa_gate_stage2_v1/reader_prompts.jsonl','results/eswa_gate_stage2_v1/reader_links.jsonl',
           'results/eswa_gate_stage2_v1/timing_recovery_qwen.jsonl','results/eswa_supplement_v2/inputs.jsonl','results/eswa_supplement_v2/scores_qwen.jsonl']
    OUT.mkdir();csvwrite(OUT/'epsilon_development_per_question.csv',exports)
    save(OUT/'epsilon.json',{'utc':utc(),'models':epsilon,'source':'old development only; no stage-2 evaluation score read for epsilon','multiplier':1.5})
    for kind in ['qwen','bge']:save_rows(OUT/f'schedule_{kind}.jsonl',tasks(kind))
    names += [str(p.relative_to(ROOT)) for p in OUT.iterdir()]
    save(OUT/'freeze.json',{'utc':utc(),'scoring_started':False,'stage1_status':'failed_86.49_percent','user_authorized_A_then_B':True,
        'files_sha256':{n:sha(ROOT/n) for n in names},'band_synthetic_cases':cases,'epsilon':epsilon,
        'requests':{kind:len(tasks(kind)) for kind in ['qwen','bge']}})
    print(__import__('json').dumps(load(OUT/'freeze.json'),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
