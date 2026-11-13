"""CPU-only selective-rescoring simulation with calibration-only threshold selection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from eswa_gate_stage0_v1 import ROOT,V2,OUT,load,rows,sha,digest,dump,table,utc,key

TAUS=[.03125,.0625,.125,.25,.5,1.]
STEPS=[.125,.25,.5]


def ranking(scores):return sorted(range(len(scores)),key=lambda i:(-scores[i],i))


class Packer:
    def __init__(self,docs,tok):
        self.blocks=['Title: '+d['title']+'\n'+d['text'] for d in docs]
        self.tok=tok;self.cache={};self.requests=0;self.misses=0
    def __call__(self,order):
        self.requests+=1;k=tuple(order[:10])
        if k in self.cache:return self.cache[k]
        selected=[];text=''
        for i in k:
            candidate='\n\n'.join(self.blocks[j] for j in selected+[i])
            if len(self.tok.encode(candidate,add_special_tokens=False))<=512:
                selected.append(i);text=candidate
        result={'selected':selected,'context':text}
        self.cache[k]=result;self.misses+=1
        return result


def gate(batch,serial,tau,packer,packing_gate=True):
    scores=list(batch);marked=set();history=[]
    while True:
        order=ranking(scores);base=packer(order);new=set();pairs=[]
        for pos in range(min(12,len(order))-1):
            a,b=order[pos:pos+2]
            if scores[a]-scores[b]>=tau:continue
            if a in marked and b in marked:continue
            changed=True
            if packing_gate:
                swapped=list(order);swapped[pos],swapped[pos+1]=swapped[pos+1],swapped[pos]
                changed=packer(swapped)['context']!=base['context']
            if changed:
                new.update({a,b}-marked);pairs.append([a,b])
        if not new:break
        assert not marked&new
        history.append({'new_indices':sorted(new),'trigger_pairs':pairs})
        marked.update(new)
        for i in new:scores[i]=serial[i]
        assert len(history)<=len(scores)
    return scores,sorted(marked),history


def refs_for(row,qasper,hotpot):
    if row['dataset']=='hotpot':
        return [set(hotpot[row['id']]['support_titles'])],'support_title_proxy'
    qa=next(q for q in qasper[row['paper_id']]['qas'] if q['question_id']==row['id'])
    refs=[]
    for annotation in qa['answers']:
        a=annotation['answer']
        if a['unanswerable']:continue
        evidence={' '.join(e.split()) for e in a['evidence'] if 'FLOAT SELECTED' not in e and e.strip()}
        if evidence:refs.append(evidence)
    return refs,'text_paragraph_recall'


def recall(row,selected,refs,kind):
    if not refs:return None
    observed={row['documents'][i]['title'] if kind=='support_title_proxy' else ' '.join(row['documents'][i]['text'].split()) for i in selected}
    return max(len(observed&g)/len(g) for g in refs if g)


def evaluate(row,model,split,serial,batch,tok,qasper,hotpot):
    packer=Packer(row['documents'],tok);ref=packer(ranking(serial));refs,kind=refs_for(row,qasper,hotpot)
    records=[];traces=[]
    for method,param in [('serial',None),('batch4',None)]+[(m,t) for m in ['gated','gap_only'] for t in TAUS]+[('top15',None)]+[(m,s) for s in STEPS for m in ['round_batch4','round_serial']]:
        start=time.perf_counter();marked=[];history=[]
        if method=='serial':scores=serial
        elif method=='batch4':scores=batch
        elif method in ['gated','gap_only']:
            scores,marked,history=gate(batch,serial,param,packer,method=='gated')
        elif method=='top15':
            scores=list(batch);marked=ranking(batch)[:15]
            for i in marked:scores[i]=serial[i]
        else:scores=[round(x/param)*param for x in (batch if method=='round_batch4' else serial)]
        pack=packer(ranking(scores));seconds=time.perf_counter()-start
        rounded_ref=packer(ranking([round(x/param)*param for x in serial])) if method.startswith('round_') else None
        record={'question_id':row['id'],'paper_id':row.get('paper_id',''),'dataset':row['dataset'],'split':split,
                'model':model,'method':method,'tau':param if method in ['gated','gap_only'] else '',
                'round_step':param if method.startswith('round_') else '', 'reference':'raw_bf16_serial',
                'candidate_count':len(batch),'serial_selected_indices':ref['selected'],'selected_indices':pack['selected'],
                'context_equal':pack['context']==ref['context'],'selected_members_equal':set(pack['selected'])==set(ref['selected']),
                'rescored_count':len(marked),'rescored_indices':sorted(marked),'rescored_fraction':len(marked)/len(batch),
                'iterations':len(history),'cpu_simulation_seconds':seconds,'gpu_reranker_seconds':'',
                'gold_evidence_recall':recall(row,pack['selected'],refs,kind),'gold_metric':kind,
                'serial_gold_evidence_recall':recall(row,ref['selected'],refs,kind),
                'context_sha256':hashlib.sha256(pack['context'].encode()).hexdigest(),
                'serial_context_sha256':hashlib.sha256(ref['context'].encode()).hexdigest(),
                'context_equal_to_rounded_serial':pack['context']==rounded_ref['context'] if rounded_ref else '',
                'members_equal_to_rounded_serial':set(pack['selected'])==set(rounded_ref['selected']) if rounded_ref else ''}
        records.append(record)
        if method in ['gated','gap_only']:traces.append({'model':model,'question_id':row['id'],'method':method,'tau':param,'rounds':history})
    return records,traces


def summarize(records):
    groups=collections.defaultdict(list)
    for r in records:groups[(r['model'],r['dataset'],r['split'],r['method'],r['tau'],r['round_step'])].append(r)
    result=[]
    for (model,dataset,split,method,tau,step),rr in groups.items():
        gold=[r['gold_evidence_recall'] for r in rr if r['gold_evidence_recall'] is not None]
        entry={'model':model,'dataset':dataset,'split':split,'method':method,'tau':tau,'round_step':step,'n':len(rr),
               'context_differences':sum(not r['context_equal'] for r in rr),'member_differences':sum(not r['selected_members_equal'] for r in rr),
               'mean_rescored_fraction':sum(r['rescored_fraction'] for r in rr)/len(rr),
               'max_rescored_fraction':max(r['rescored_fraction'] for r in rr),
               'mean_rescored_candidates':sum(r['rescored_count'] for r in rr)/len(rr),
               'max_rescored_candidates':max(r['rescored_count'] for r in rr),
               'gold_recall_mean':sum(gold)/len(gold) if gold else None,'gold_recall_n':len(gold),
               'gold_metric':rr[0]['gold_metric'], 'cpu_simulation_seconds_sum':sum(r['cpu_simulation_seconds'] for r in rr),
               'rounded_reference_context_differences':sum(not r['context_equal_to_rounded_serial'] for r in rr) if method.startswith('round_') else '',
               'rounded_reference_member_differences':sum(not r['members_equal_to_rounded_serial'] for r in rr) if method.startswith('round_') else ''}
        result.append(entry)
    for r in result:
        a=next(x for x in result if x['model']==r['model'] and x['dataset']==r['dataset'] and x['split']==r['split'] and x['method']=='batch4')
        baseline=a['context_differences']
        r['baseline_context_differences']=baseline
        r['context_difference_reduction']=1-r['context_differences']/baseline if baseline else None
        r['meets_90pct_reduction']=(r['context_differences']*10<=baseline) if baseline else False
        r['strictly_exceeds_90pct_reduction']=(r['context_differences']*10<baseline) if baseline else False
        r['meets_20pct_mean_budget']=r['mean_rescored_fraction']<=.2+1e-12
        r['meets_joint_gate']=r['meets_90pct_reduction'] and r['meets_20pct_mean_budget']
    return result


def choose(summary,model,method):
    eligible=[r for r in summary if r['model']==model and r['method']==method and r['split']=='calibration']
    passing=[r for r in eligible if r['meets_joint_gate']]
    budget=[r for r in eligible if r['meets_20pct_mean_budget']]
    if passing:
        picked=min(passing,key=lambda r:(r['mean_rescored_fraction'],r['context_differences'],r['tau']));rule='feasible_minimum_cost'
    elif budget:
        picked=min(budget,key=lambda r:(r['context_differences'],r['mean_rescored_fraction'],r['tau']));rule='no_joint_feasible_tau_best_under_budget'
    else:
        picked=min(eligible,key=lambda r:(r['mean_rescored_fraction'],r['context_differences'],r['tau']));rule='no_tau_under_budget'
    return {'model':model,'method':method,'selected_tau':picked['tau'],'rule':rule,'calibration_result':picked}


def select_main(summary,selection):
    out=[]
    for row in summary:
        method=row['method'];r=dict(row)
        tau=next(s['selected_tau'] for s in selection if s['model']==r['model'] and s['method']=='gated')
        gap=next(s['selected_tau'] for s in selection if s['model']==r['model'] and s['method']=='gap_only')
        if method=='gated' and r['tau']!=tau:continue
        if method=='gap_only':
            if r['tau']==tau:r['method']='gap_only_same_tau'
            elif r['tau']==gap:r['method']='gap_only_independent_tau'
            else:continue
        r['selected_for_primary_gate']=method=='gated'
        out.append(r)
    return out


def selftest():
    # Distinct length-limited synthetic blocks; hand-derived boundary cases.
    class Synthetic:
        def __call__(self,order):
            selected=list(order[:10]);return {'selected':selected,'context':str(selected)}
    scores=list(reversed([float(i) for i in range(15)]));serial=list(scores)
    _,marked,_=gate(scores,serial,1.,Synthetic(),False)
    assert marked==[], 'The threshold is strict, not <='
    scores[9]=5.1;scores[10]=5.09
    serial[9]=5.08;serial[10]=5.11
    new,marked,hist=gate(scores,serial,.03125,Synthetic(),True)
    assert {9,10}<=set(marked) and len(marked)==len(set(marked))
    assert set(Synthetic()(ranking(new))['selected'])==set(Synthetic()(ranking(serial))['selected'])
    assert [round(x/.125)*.125 for x in [.0625,.1875,-.0625,-.1875]]==[0.,.25,0.,-.25]


def main():
    assert os.name!='nt' and not (ROOT/'logs/research.pause').exists()
    stage0=load(OUT/'stage0/audit.json');assert stage0['passed'] and not stage0['stop']
    assert sha(V2/'inputs.jsonl')==stage0['source_hashes']['v2_inputs']
    base=OUT/'stage1';assert not base.exists();base.mkdir()
    result={'stage':1,'started_utc':utc(),'state':'running'}
    selftest()
    original=rows(V2/'inputs.jsonl');qas=[r for r in original if r['dataset']=='qasper'];hot=[r for r in original if r['dataset']=='hotpot']
    assert [r['id'] for r in qas]==stage0['qasper_ordered_ids']
    calibration=qas[:64];test=qas[64:];assert len(calibration)==64 and len(test)==128
    assert not {r['id'] for r in calibration}&{r['id'] for r in test}
    freeze={'utc':utc(),'stage0_audit_sha256':sha(OUT/'stage0/audit.json'),'code_sha256':sha(__file__),
            'protocol_sha256':sha(ROOT/'research/eswa_gate_staged_protocol_20261110_v1.md'),
            'calibration_ids':[r['id'] for r in calibration],'test_ids':[r['id'] for r in test],
            'taus':TAUS,'round_steps':STEPS,'gpu_inference':False,'selection_uses_test':False}
    dump(base/'freeze.json',freeze)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen--Qwen3-8B'),local_files_only=True)
    qasper=load(ROOT/'data/eswa_supplement_v2/qasper-dev-v0.3.json');hotgold={r['id']:r for r in rows(ROOT/'data/stage22_reserved64/labels.jsonl')}
    scores={}
    for model in ['qwen','minilm']:
        assert sha(V2/f'scores_{model}.jsonl')==load(V2/f'score_runtime_{model}.json')['scores_sha256']
        for r in rows(V2/f'scores_{model}.jsonl'):
            if r['rep']==0 and r['arm'] in ['bf16_serial','bf16_batch4']:scores[model,r['id'],r['arm']]=r
    all_records=[];all_traces=[]
    try:
        for split,population in [('calibration',calibration),('test',test),('descriptive_hotpot',hot)]:
            for model in ['qwen','minilm']:
                for i,row in enumerate(population):
                    assert not (ROOT/'logs/research.pause').exists()
                    a=scores[model,row['id'],'bf16_serial'];b=scores[model,row['id'],'bf16_batch4']
                    r,tr=evaluate(row,model,split,a['scores'],b['scores'],tok,qasper,hotgold)
                    # Check baseline equality again at the simulation boundary.
                    for rec in r:
                        if rec['method'] in ['serial','batch4']:
                            saved=a if rec['method']=='serial' else b
                            assert rec['selected_indices']==saved['packs']['512']['selected']
                            assert rec['context_sha256']==hashlib.sha256(saved['packs']['512']['context'].encode()).hexdigest()
                    all_records.extend(r);all_traces.extend(tr)
                    if (i+1)%16==0:print(json.dumps({'split':split,'model':model,'questions':i+1,'of':len(population)}),flush=True)
            if split=='calibration':
                calibration_summary=summarize(all_records)
                table(base/'calibration_grid.csv',calibration_summary)
                selection=[choose(calibration_summary,m,k) for m in ['qwen','minilm'] for k in ['gated','gap_only']]
                dump(base/'tau_selection.json',{'utc':utc(),'selection':selection,'test_scored':False,'test_labels_used':False})
                print(json.dumps({'thresholds_frozen':selection}),flush=True)
        summary=summarize(all_records);main_table=select_main(summary,selection)
        qwen_gate=next(r for r in main_table if r['model']=='qwen' and r['split']=='test' and r['method']=='gated')
        top15=next(r for r in main_table if r['model']=='qwen' and r['split']=='test' and r['method']=='top15')
        passed=qwen_gate['meets_joint_gate'] or top15['meets_joint_gate']
        exact_boundary=any(r['meets_joint_gate'] and not r['strictly_exceeds_90pct_reduction'] for r in [qwen_gate,top15])
        table(base/'per_question_all.csv',all_records);table(base/'all_results.csv',summary);table(base/'selected_results.csv',main_table)
        keys={(r['model'],r['split'],r['method'],r['tau'],r['round_step']) for r in main_table}
        selected_records=[]
        for r in all_records:
            name=r['method']
            for variant in ([name] if name!='gap_only' else ['gap_only_same_tau','gap_only_independent_tau']):
                if (r['model'],r['split'],variant,r['tau'],r['round_step']) in keys:selected_records.append({**r,'method':variant})
        table(base/'per_question_selected.csv',selected_records)
        with (base/'iteration_traces.jsonl').open('x',encoding='utf-8') as f:
            for r in all_traces:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        result.update(state='passed' if passed and not exact_boundary else 'stopped',passed=passed and not exact_boundary,
                      stop=not passed or exact_boundary,finished_utc=utc(),selected_taus=selection,
                      qwen_test_gated=qwen_gate,qwen_test_top15=top15,ambiguous_exact_90pct_boundary=exact_boundary,
                      all_rows=len(all_records),selected_rows=len(selected_records),summary_cells=len(summary),
                      stop_reason=None if passed and not exact_boundary else 'Neither selected Qwen gating nor top15 achieves both the required context reduction and <=20% mean rescoring.' if not passed else 'Exactly 90% reduction requires clarification of the boundary before continuing.',
                      stage2_started=False,stage3_started=False,manuscript_written=False,
                      timing_scope='CPU simulation with exact pack memoization; not measured GPU rescoring latency.')
    except BaseException as exc:
        result.update(state='failed',passed=False,stop=True,error=repr(exc),finished_utc=utc())
        raise
    finally:dump(base/'decision.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
