"""Supplement namespace; never mutates the frozen stage-2 sources or observations."""
import csv
from eswa_stage2_common import ROOT, BASE as STAGE2, DATA, CFG as OLD_CFG, load, sha, utc, save, save_rows, read_jsonl, digest, check

CFG = load(ROOT/'configs/eswa_extension_20261111_v1.json')
OUT = ROOT/CFG['output']

def csvwrite(path, rows):
    rows=list(rows);fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        import json
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict,tuple)) else v for k,v in r.items()})

def identity(r):
    return (r['model'],r['id'],r['method'],r['batch_size'],r['rep'],r.get('seed'))

def tasks(kind):
    rows=[r for r in read_jsonl(STAGE2/'inputs.jsonl') if r['role']=='evaluation'];ans=[]
    cells=[(m,b,'scan',None) for b in CFG['batches'] for m in ['native','gate_v2','top15']]
    cells += [('band',b,'band',None) for b in CFG['band_batches']]
    for rep in range(2):
        for ix,row in (list(enumerate(rows)) if rep==0 else list(reversed(list(enumerate(rows))))):
            order=cells[ix%len(cells):]+cells[:ix%len(cells)]
            if rep:order=list(reversed(order))
            for method,size,role,seed in order:ans.append(dict(model=kind,id=row['id'],method=method,batch_size=size,role=role,rep=rep,seed=seed))
    for row in rows:ans.append(dict(model=kind,id=row['id'],method='fp32_math_serial',batch_size=1,role='precision',rep=0,seed=None))
    if kind=='qwen':
        for seed in CFG['random_seeds']:
            for ix,row in enumerate(rows):
                for method in (['random_native','random_gate_v2'] if ix%2==0 else ['random_gate_v2','random_native']):
                    ans.append(dict(model=kind,id=row['id'],method=method,batch_size=8,role='random',rep=0,seed=seed))
    assert len({identity(r) for r in ans})==len(ans)
    return ans

def band_indices(scores,epsilon):
    from eswa_gate_methods_v2 import order
    assert len(scores)>0 and epsilon>=0
    kth=scores[order(scores)[min(10,len(scores))-1]]
    return [i for i,b in enumerate(scores) if b>=kth-2*epsilon],kth

def verify():
    check();f=load(OUT/'freeze.json')
    for path,want in f['files_sha256'].items():assert sha(ROOT/path)==want,path
    from run_eswa_stage2_v1 import verify_freeze
    verify_freeze()
    return f
