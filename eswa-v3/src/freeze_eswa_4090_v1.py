"""CPU-only integrity gate before any RTX4090 calibration scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast,collections,json,random
from eswa_4090_common_v1 import *

def dependencies(start):
    seen=set();pending=list(start)
    while pending:
        name=pending.pop()
        if name in seen:continue
        path=ROOT/'src'/name;assert path.exists(),path;seen.add(name)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            modules=([node.module] if isinstance(node,ast.ImportFrom) and node.module else [a.name for a in node.names] if isinstance(node,ast.Import) else [])
            for module in modules:
                base=module.split('.')[0]+'.py'
                if (ROOT/'src'/base).exists():pending.append(base)
    return sorted('src/'+n for n in seen)

def main():
    assert os.name!='nt';assert not (OUT/'freeze.json').exists()
    assert not any(OUT.glob('calibration_*.jsonl'))
    for p,h in load(OUT/'rules_freeze.json')['files_sha256'].items():assert sha(ROOT/p)==h
    for p,h in load(OUT/'source_transfer_manifest.json')['files_sha256'].items():assert sha(ROOT/p)==h,('source input transfer mismatch',p)
    assert sha(OUT/'inputs.jsonl')==load(OUT/'candidate_manifest.json')['inputs_sha256']
    inputs=read_jsonl(OUT/'inputs.jsonl');sampling=load(OUT/'sampling.json')
    assert len(inputs)==456 and len({r['id'] for r in inputs})==456
    assert collections.Counter(r['role']+'/'+r['dataset'] for r in inputs)=={'evaluation/hotpot':200,'evaluation/qasper':192,'calibration/hotpot':32,'calibration/qasper':32}
    assert len({r['paper_id'] for r in inputs if r['dataset']=='qasper'})==224
    assert not set(sampling['evaluation_ids'])&set(sampling['calibration_ids'])
    cross=read_jsonl(DATA/'cross_hardware_inputs.jsonl');assert len(cross)==392 and not {r['id'] for r in cross}&{r['id'] for r in inputs}
    assert not {r['paper_id'] for r in cross if r['dataset']=='qasper'}&{r['paper_id'] for r in inputs if r['dataset']=='qasper'}
    from eswa_extension_common_v1 import band_indices
    # Bound implication with ties, K>N, threshold equality, and arbitrary original indices.
    rng=random.Random(20261112);checks=0
    for n in [1,3,10,11,16,100]:
        for epsilon in [0,.125,.1875,.9375]:
            for _ in range(100):
                serial=[rng.randrange(-10,11)/8 for i in range(n)];batch=[s+rng.choice([-1,0,1])*epsilon for s in serial]
                subset,kth=band_indices(batch,epsilon);expected=sorted(range(n),key=lambda i:(-serial[i],i))[:10]
                got=sorted(subset,key=lambda i:(-serial[i],i))[:10];assert got==expected;checks+=1
    plans={};task_rows={};total=0
    for model in ['qwen','bge']:
        for phase in ['calibration','evaluation','cross']:
            plan=tasks(model,phase);name=f'{phase}_{model}';plans[name]=len(plan);total+=len(plan);task_rows[name]=plan
    assert total==26896 and plans=={'calibration_qwen':512,'evaluation_qwen':12544,'cross_qwen':392,'calibration_bge':512,'evaluation_bge':12544,'cross_bge':392}
    source=load(DATA/'stage2_asset_manifest.json');assets=source['assets_sha256']
    shared=['src/run_eswa_stage2_v1.py','src/eswa_stage2_common.py','src/eswa_gate_methods_v2.py','src/eswa_stage2_metrics.py','configs/eswa_gate_stage2_v1.json']
    for p in shared:assert sha(ROOT/p)==source['files_sha256'][p],('stage2 implementation changed',p)
    for p,h in assets.items():assert sha(ROOT/p)==h,('asset mismatch',p)
    names=dependencies(['run_eswa_4090_v1.py','freeze_eswa_4090_v1.py','analyze_eswa_4090_v1.py'])
    names+=['configs/eswa_4090_validation_v1.json','configs/eswa_gate_stage2_v1.json','configs/eswa_extension_20261111_v1.json','configs/qwen512_v1.json','research/eswa_4090_validation_protocol_v1.md','run_eswa_4090_queue.sh','src/prepare_eswa_4090_v1.py','src/launch_eswa_4090_ready_v1.py']
    names+=[str(p.relative_to(ROOT)) for p in DATA.glob('*') if p.is_file()]
    names+=[str(OUT.relative_to(ROOT))+'/'+n for n in ['inputs.jsonl','rules_freeze.json','sampling.json','candidate_manifest.json','source_transfer_manifest.json']]
    names+=[str((OUT/f'plan_{name}.jsonl').relative_to(ROOT)) for name in task_rows]
    import torch,transformers,numpy,peft,platform,importlib.metadata,qwen_student
    assert torch.__version__=='2.5.1+cu121' and transformers.__version__=='4.57.6';assert not torch.cuda.is_initialized()
    env={'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'numpy':numpy.__version__,'peft':peft.__version__,'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}}
    for name,plan in task_rows.items():save_rows(OUT/f'plan_{name}.jsonl',plan)
    save(OUT/'freeze.json',{'utc':utc(),'rules_freeze_sha256':sha(OUT/'rules_freeze.json'),'scoring_started':False,'files_sha256':{n:sha(ROOT/n) for n in sorted(set(names))},'assets_sha256':assets,'environment':env,'synthetic_band_bound_checks':checks,'planned_requests':plans,'planned_total':total})
    print(json.dumps({'state':'frozen','freeze_sha256':sha(OUT/'freeze.json'),'planned':total,'band_checks':checks}),flush=True)

if __name__=='__main__':main()
