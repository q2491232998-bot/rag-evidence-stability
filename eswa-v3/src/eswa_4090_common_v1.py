"""Dedicated validation namespace; historical configurations remain immutable."""
import os,shutil,time
from eswa_stage2_common import ROOT,load,save,save_rows,read_jsonl,sha,utc,digest
C=load(ROOT/'configs/eswa_4090_validation_v1.json');OUT=ROOT/C['output'];DATA=ROOT/C['data']
def identity(t):return t['model'],t['id'],t['method'],t['batch_size'],t['rep'],t.get('seed')
def tasks(model,phase):
    if phase=='cross':
        return [dict(model=model,id=r['id'],method='serial',batch_size=1,rep=0,seed=None,role='cross') for r in read_jsonl(DATA/'cross_hardware_inputs.jsonl')]
    rows=[r for r in read_jsonl(OUT/'inputs.jsonl') if r['role']==('calibration' if phase=='calibration' else 'evaluation')]
    cells=[('serial',1)]+[(m,b) for b in C['batches'] for m in (['native'] if phase=='calibration' else ['native','gate_v2','top15','band_transferred','band_recalibrated'])]
    answer=[]
    for rep in range(C['repetitions']):
        for ix,row in (list(enumerate(rows)) if not rep else list(reversed(list(enumerate(rows))))):
            ordered=cells[ix%len(cells):]+cells[:ix%len(cells)]
            if rep:ordered=list(reversed(ordered))
            for method,size in ordered:answer.append(dict(model=model,id=row['id'],method=method,batch_size=size,rep=rep,seed=None,role=phase))
    assert len({identity(t) for t in answer})==len(answer);return answer
def check_deadline():
    assert os.name!='nt','Remote computation only'
    assert not (ROOT/'logs/research.pause').exists(),'Pause marker'
    assert shutil.disk_usage(ROOT).free>=C['minimum_free_bytes'],'Disk headroom'
    assert time.time()<load(OUT/'queue_started.json')['deadline_epoch'],'24-hour window expired; no automatic extension'
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<C['maximum_result_bytes'],'Result size cap'
def verify():
    f=load(OUT/'freeze.json')
    for p,h in f['files_sha256'].items():assert sha(ROOT/p)==h,p
    return f
