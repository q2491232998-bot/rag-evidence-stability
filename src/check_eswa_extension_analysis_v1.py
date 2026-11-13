"""CPU validation against existing observed stage-2 traces; no new evaluation fitting."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from eswa_extension_common_v1 import *
from analyze_eswa_extension_v1 import pack, independent_triggers, sort
from run_eswa_extension_v1 import permutation

def main():
    verify()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen--Qwen3-8B'),local_files_only=True)
    inputs={r['id']:r for r in read_jsonl(STAGE2/'inputs.jsonl')};checks=0
    for model in ['qwen','bge']:
        records=read_jsonl(STAGE2/f'requests_{model}.jsonl')
        for data in ['qasper','hotpot']:
            chosen=[r for r in records if r['dataset']==data and r['method']=='gate_v2' and r['rep']==0][:8]
            for r in chosen:
                row=inputs[r['id']];values=list(r['initial_scores']);seen=set();cache={}
                for rd in r['rounds']:
                    fresh,pairs=independent_triggers(row,values,seen,r['tau'],tok,cache)
                    assert fresh==rd['indices'] and pairs==rd['trigger_pairs']
                    for i,v in rd['actual_serial_scores'].items():values[int(i)]=v
                    seen.update(rd['indices'])
                assert independent_triggers(row,values,seen,r['tau'],tok,cache)[0]==[]
                assert pack(row,sort(values),tok)==(r['selected'],r['context'],r['tokens']);checks+=1
    for n in [10,11,100,192]:
        pp=[permutation('fixed-id',n,seed) for seed in CFG['random_seeds']]
        assert all(sorted(p)==list(range(n)) and p!=list(range(n)) for p in pp)
        assert len({tuple(p) for p in pp})==3
        assert pp[0]==permutation('fixed-id',n,CFG['random_seeds'][0])
    save(OUT/'analysis_preflight.json',{'utc':utc(),'state':'passed','existing_gate_traces_replayed':checks,'random_permutation_contract':'passed','new_scoring':False})
    print(load(OUT/'analysis_preflight.json'),flush=True)

if __name__=='__main__':main()
