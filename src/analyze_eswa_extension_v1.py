"""Remote CPU audits, complete exports, and CSV-derived batch/answer motivation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections
import csv
import importlib.util
import itertools
import math
import statistics
from eswa_extension_common_v1 import *
from eswa_stage2_metrics import *

def avg(x):
    x=[v for v in x if v is not None];return sum(x)/len(x) if x else None

def old_key(r):return r['model'],r['id'],r['method'],int(r['batch_size']),int(r['rep'])

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def pack(row,ranking,tok):
    selected=[];text=''
    for i in ranking[:10]:
        d=row['documents'][i];proposal=(text+'\n\n' if selected else '')+'Title: '+d['title']+'\n'+d['text']
        if len(tok.encode(proposal,add_special_tokens=False))<=512:selected.append(i);text=proposal
    return selected,text,len(tok.encode(text,add_special_tokens=False))

def sort(scores):return sorted(range(len(scores)),key=lambda i:(-scores[i],i))

def independent_triggers(row,values,seen,tau,tok,cache):
    ranking=sort(values)
    def cached(r):
        k=tuple(r[:10])
        if k not in cache:cache[k]=pack(row,r,tok)[1]
        return cache[k]
    original=cached(ranking);pairs=[];fresh=set()
    for a,b in itertools.combinations(range(min(12,len(ranking))),2):
        i,j=ranking[a],ranking[b]
        if values[i]-values[j]>=tau or (i in seen and j in seen):continue
        swapped=list(ranking);swapped[a],swapped[b]=swapped[b],swapped[a]
        if cached(swapped)!=original:pairs.append([i,j]);fresh.update({i,j}-seen)
    return sorted(fresh),pairs

def main():
    freeze=verify();assert load(OUT/'queue_finished.json')['state']=='gpu_complete_analysis_pending'
    assert load(OUT/'reader_runtime.json')['state']=='complete'
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen--Qwen3-8B'),local_files_only=True)
    spec=importlib.util.spec_from_file_location('qasper',DATA/'qasper_evaluator.py');official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    refs=official.get_answers_and_evidence(load(DATA/'qasper-test-v0.3.json'),text_evidence_only=True)
    hot={r['_id']:r for r in read_jsonl(DATA/'hotpot_labels.jsonl')};rows={r['id']:r for r in read_jsonl(STAGE2/'inputs.jsonl')}
    oldrecords={};records={}
    for kind in ['qwen','bge']:
        oldrecords.update({old_key(r):r for r in read_jsonl(STAGE2/f'requests_{kind}.jsonl')})
        runtime=load(OUT/f'runtime_{kind}.json');assert runtime['state']=='complete' and runtime['parameters_unchanged'] and runtime['buffers_unchanged']
        assert sha(OUT/f'requests_{kind}.jsonl')==runtime['records_sha256']
        new=read_jsonl(OUT/f'requests_{kind}.jsonl');assert len(new)==len(tasks(kind))
        assert {identity(r) for r in new}=={identity(r) for r in tasks(kind)}
        records.update({identity(r):r for r in new})
    assert len(records)==26656
    previous=read_csv(STAGE2/'analysis/per_question_repeats.csv');oldcsv={old_key(r):r for r in previous}
    answers={r['key']:r for r in read_jsonl(OUT/'answers.jsonl')};links={identity(r):r['key'] for r in read_jsonl(OUT/'reader_links.jsonl')}
    assert sha(OUT/'answers.jsonl')==load(OUT/'reader_runtime.json')['answers_sha256']
    eps=load(OUT/'epsilon.json')['models'];cache={};gatecache={};counts=collections.Counter();exports=[];randomrows=[];allrows={}
    for ident,r in records.items():
        row=rows[r['id']];values=list(r['initial_scores']);seen=set();isband=r['method']=='band';selected_band=None
        reference=oldrecords[r['model'],r['id'],'serial',1,r['rep']]
        assert r['candidate_count']==len(values)==len(row['documents']) and all(math.isfinite(v) for v in values)
        if isband:
            kth=values[sort(values)[min(10,len(values))-1]];selected_band=[i for i,v in enumerate(values) if v>=kth-2*r['epsilon']]
            assert selected_band==r['band_indices'] and kth==r['batch_kth_score']
        gcache=gatecache.setdefault(r['id'],{})
        mismatched=0
        for iteration,rd in enumerate(r['rounds']):
            actual={int(i):v for i,v in rd['actual_serial_scores'].items()};assert set(rd['indices'])==set(actual) and not seen&set(actual)
            if 'gate_v2' in r['method']:
                fresh,pairs=independent_triggers(row,values,seen,r['tau'],tok,gcache);assert fresh==rd['indices'] and pairs==rd['trigger_pairs']
            else:
                assert iteration==0 and rd['indices']==(selected_band if isband else sort(values)[:15])
            for i,s in actual.items():
                counts['actual_serial_values']+=1;mismatched+=int(s!=reference['scores'][i]);values[i]=s
            seen.update(actual)
        if 'gate_v2' in r['method']:assert independent_triggers(row,values,seen,r['tau'],tok,gcache)[0]==[]
        assert values==r['scores'] and sorted(seen)==r['rescored_indices'] and len(seen)==r['rescored_count']
        assert len(seen)/len(values)==r['rescored_fraction']
        ranking=sorted(selected_band,key=lambda i:(-values[i],i))[:10] if isband else sort(values)
        assert ranking==r['order'];pk=(r['id'],tuple(ranking[:10]))
        if pk not in cache:cache[pk]=pack(row,ranking,tok)
        chosen,text,tokens=cache[pk];assert (chosen,text,tokens)==(r['selected'],r['context'],r['tokens'])
        counts['independent_packs']+=1;counts['actual_serial_values_different']+=mismatched
        assert r['gpu_isolation_observed'] and r['total_seconds']>0
        if r['seed'] is not None:
            from run_eswa_extension_v1 import permutation
            assert r['grouping_permutation']==permutation(r['id'],len(values),r['seed'])
        if r['role']=='random':
            randomrows.append({k:r[k] for k in ['model','dataset','id','method','batch_size','seed']}|{'context_equal_serial':text==reference['context'],
                'context_sha256':digest(text),'serial_context_sha256':digest(reference['context'])})
            continue
        paired=[records[r['model'],r['id'],r['method'],r['batch_size'],rep,None] for rep in ([0] if r['role']=='precision' else [0,1])]
        median=statistics.median(v['total_seconds'] for v in paired)
        serial_csv=oldcsv[r['model'],r['id'],'serial',1,r['rep']];serial_time=float(serial_csv['two_repeat_median_seconds'])
        pred=answers[links[ident]];docs=[row['documents'][i] for i in chosen]
        if row['dataset']=='qasper':
            metric=qasper_evidence(docs,refs[r['id']],official)
            metric.update(answer_f1=max(official.token_f1_score(pred['answer'],a['answer']) for a in refs[r['id']]),
                          answer_em=max(float(official.normalize_answer(pred['answer'])==official.normalize_answer(a['answer'])) for a in refs[r['id']]))
        else:metric={**hotpot_evidence(docs,hot[r['id']]),**hotpot_answer(pred['answer'],hot[r['id']]['answer'])}
        errors=[abs(a-b) for a,b in zip(r['initial_scores'],reference['scores'])]
        over=sum(d>eps[r['model']]['epsilon'] for d in errors) if isband else None
        if isband and not over and not mismatched:
            assert ranking==reference['order'][:10] and text==reference['context'],'Conditional band theorem violation'
            counts['band_bound_holds_and_verified']+=1
        result={k:r[k] for k in ['model','dataset','id','paper_id','method','batch_size','rep','seed','role','tau','epsilon','candidate_count',
                    'selected','rescored_indices','rescored_count','rescored_fraction','total_seconds','forward_seconds','gate_check_seconds','gpu_isolation_observed','attempt']}
        result.update(serial_selected=reference['selected'],context_equal_serial=text==reference['context'],members_equal_serial=set(chosen)==set(reference['selected']),
            context_sha256=digest(text),serial_context_sha256=digest(reference['context']),context_tokens=tokens,
            timed_repetitions=len(paired),query_median_seconds=median,serial_stage2_median_seconds=serial_time,time_ratio_to_serial=median/serial_time,
            repeat_context_identical=all(v['context']==r['context'] for v in paired),repeat_scores_identical=all(v['scores']==r['scores'] for v in paired),
            actual_serial_rescores_different=mismatched,epsilon_exceeded_candidates=over,maximum_initial_error_to_bf16_serial=max(errors),
            all_candidate_errors_within_epsilon=(over==0) if isband else None,band_selected_count=len(selected_band) if isband else None,
            answer=pred['answer'],answer_equal_serial=pred['answer']==serial_csv['answer'],normalized_answer_equal_serial=answer_norm(pred['answer'])==answer_norm(serial_csv['answer']),
            reader_cap_hit=pred['cap_hit'],truncated_candidates=sum(r['truncated']),**metric)
        exports.append(result);allrows[ident]=result
    # Compare initial native scores without hiding possible cross-session variation.
    for ident,r in records.items():
        if r['method'] in ['gate_v2','top15','band']:
            native=(records[r['model'],r['id'],'native',r['batch_size'],r['rep'],None] if r['batch_size']!=4 else oldrecords[r['model'],r['id'],'batch4',4,r['rep']])
            counts['initial_vectors_compared']+=1;counts['initial_vectors_different_from_native']+=int(r['initial_scores']!=native['scores'])
        if r['method']=='random_gate_v2':
            native=records[r['model'],r['id'],'random_native',8,0,r['seed']]
            counts['random_initial_vectors_compared']+=1;counts['random_initial_vectors_different']+=int(r['initial_scores']!=native['scores'])
    groups=collections.defaultdict(list)
    for r in exports:groups[r['model'],r['dataset'],r['method'],r['batch_size'],r['rep']].append(r)
    summary=[]
    for (model,data,method,size,rep),rr in sorted(groups.items()):
        n=len(rr);diff=sum(not r['context_equal_serial'] for r in rr)
        summary.append(dict(model=model,dataset=data,method=method,batch_size=size,rep=rep,n=n,context_differences=diff,
            member_differences=sum(not r['members_equal_serial'] for r in rr),mean_rescored_fraction=avg(r['rescored_fraction'] for r in rr),
            max_rescored_fraction=max(r['rescored_fraction'] for r in rr),gold_recall=avg(r['gold_evidence_recall'] for r in rr),
            gold_recall_n=sum(r['gold_evidence_recall'] is not None for r in rr),answer_f1=avg(r['answer_f1'] for r in rr),answer_em=avg(r['answer_em'] for r in rr),
            answer_differences=sum(not r['answer_equal_serial'] for r in rr),normalized_answer_differences=sum(not r['normalized_answer_equal_serial'] for r in rr),
            sum_query_median_seconds=sum(r['query_median_seconds'] for r in rr),aggregate_time_ratio_to_serial=sum(r['query_median_seconds'] for r in rr)/sum(r['serial_stage2_median_seconds'] for r in rr),
            epsilon_exceeded_candidates=sum(r['epsilon_exceeded_candidates'] or 0 for r in rr) if method=='band' else None,
            epsilon_exceeded_questions=sum(bool(r['epsilon_exceeded_candidates']) for r in rr) if method=='band' else None,
            candidate_observations=sum(r['candidate_count'] for r in rr),max_error_to_bf16_serial=max(r['maximum_initial_error_to_bf16_serial'] for r in rr),
            repeat_context_changes=sum(not r['repeat_context_identical'] for r in rr),repeat_score_changes=sum(not r['repeat_scores_identical'] for r in rr),
            zero_change_one_sided95_upper=1-.05**(1/n) if diff==0 else None,timed_repetitions=rr[0]['timed_repetitions']))
    randomsummary=[]
    for data,seed,method in itertools.product(['qasper','hotpot'],CFG['random_seeds'],['random_native','random_gate_v2']):
        rr=[r for r in randomrows if (r['dataset'],r['seed'],r['method'])==(data,seed,method)];diff=sum(not r['context_equal_serial'] for r in rr)
        randomsummary.append(dict(model='qwen',dataset=data,seed=seed,method=method,batch_size=8,n=len(rr),context_differences=diff,
            zero_change_one_sided95_upper=1-.05**(1/len(rr)) if diff==0 else None))
    precision=[]
    for r in records.values():
        if r['role']!='precision':continue
        for rep in [0,1]:
            b=oldrecords[r['model'],r['id'],'fp32_math_batch4',4,rep];bcsv=oldcsv[old_key(b)];a=allrows[identity(r)]
            precision.append(dict(model=r['model'],dataset=r['dataset'],id=r['id'],batch_rep=rep,serial_repetitions=1,
                context_equal_fp32_serial=b['context']==r['context'],fp32_serial_equal_bf16_serial=a['context_equal_serial'],
                members_equal_fp32_serial=set(b['selected'])==set(r['selected']),fp32_serial_selected=r['selected'],fp32_batch4_selected=b['selected'],
                fp32_serial_seconds=r['total_seconds'],fp32_batch4_median_seconds=float(bcsv['two_repeat_median_seconds']),bf16_serial_median_seconds=a['serial_stage2_median_seconds'],
                score_candidates_different=sum(x!=y for x,y in zip(b['scores'],r['scores'])),maximum_score_difference=max(abs(x-y) for x,y in zip(b['scores'],r['scores']))))
    precisionsummary=[]
    for model,data,rep in itertools.product(['qwen','bge'],['qasper','hotpot'],[0,1]):
        rr=[r for r in precision if (r['model'],r['dataset'],r['batch_rep'])==(model,data,rep)];den=sum(r['bf16_serial_median_seconds'] for r in rr)
        precisionsummary.append(dict(model=model,dataset=data,batch_rep=rep,n=len(rr),context_differences_fp32_batch4_vs_fp32_serial=sum(not r['context_equal_fp32_serial'] for r in rr),
            context_differences_fp32_serial_vs_bf16_serial=sum(not r['fp32_serial_equal_bf16_serial'] for r in rr),
            fp32_serial_time_ratio_to_bf16_serial=sum(r['fp32_serial_seconds'] for r in rr)/den,fp32_batch4_time_ratio_to_bf16_serial=sum(r['fp32_batch4_median_seconds'] for r in rr)/den,
            score_candidates_different=sum(r['score_candidates_different'] for r in rr),maximum_score_difference=max(r['maximum_score_difference'] for r in rr),
            serial_repetitions=1,batch_repetitions_for_time=2))
    contaminated=read_jsonl(OUT/'contaminated_attempts.jsonl') if (OUT/'contaminated_attempts.jsonl').exists() else []
    contaminated_export=[]
    for r in contaminated:
        matched=records.get(identity(r));contaminated_export.append({k:r[k] for k in ['model','id','method','batch_size','rep','seed','role','attempt','total_seconds','selected','rescored_count']}|
            {'context_sha256':digest(r['context']),'matches_effective_context':r['context']==matched['context'] if matched else None,
             'matches_effective_scores':r['scores']==matched['scores'] if matched else None,'external_gpu_observations':r['external_gpu_observations']})
    occupancy=read_jsonl(OUT/'gpu_occupancy.jsonl');external=[r for r in occupancy if any(not p['own'] for p in r['processes'])]
    occupation=[{'monitor_interval_seconds':.5,'observations':len(occupancy),'external_observation_records':len(external),
        'speech_service_observation_records':sum(any(p['speech_service_detected'] and not p['own'] for p in r['processes']) for r in occupancy),
        'contaminated_attempts':len(contaminated),'effective_requests':len(records),'effective_requests_with_detected_foreign_process':sum(not r['gpu_isolation_observed'] for r in records.values()),
        'claim':'sampling evidence, not exclusion of subinterval activity'}]
    out=OUT/'analysis';out.mkdir(exist_ok=False)
    csvwrite(out/'per_question_repeats.csv',exports);csvwrite(out/'per_question_medians.csv',[r for r in exports if r['rep']==0])
    csvwrite(out/'all_results.csv',summary);csvwrite(out/'main_results.csv',[r for r in summary if r['rep']==0]);csvwrite(out/'random_grouping_per_question.csv',randomrows)
    csvwrite(out/'random_grouping_summary.csv',randomsummary);csvwrite(out/'precision_per_question.csv',precision);csvwrite(out/'precision_summary.csv',precisionsummary)
    csvwrite(out/'gpu_occupancy_summary.csv',occupation)
    if contaminated_export:csvwrite(out/'contaminated_attempts.csv',contaminated_export)
    # Motivation is recalculated from the emitted CSV, not from prose or an old tally.
    motivation=[];motivation_summary=[]
    for source,rr,sizes in [('stage2_qwen_qasper',previous,[1,2,4,8,16,32]),('extension_native',read_csv(out/'per_question_repeats.csv'),[2,8,16,32])]:
        chosen=[r for r in rr if int(r['rep'])==0 and ((source=='stage2_qwen_qasper' and r['model']=='qwen' and r['dataset']=='qasper' and r['method'] in ['serial','batch4','batch_scan']) or (source=='extension_native' and r['method']=='native'))]
        grouped=collections.defaultdict(list)
        for r in chosen:grouped[r['model'],r['dataset'],r['id']].append(r)
        for (model,data,qid),parts in grouped.items():
            assert sorted(int(r['batch_size']) for r in parts)==sizes
            value=dict(source=source,model=model,dataset=data,id=qid,batch_sizes=sizes,context_varies=len({r['context_sha256'] for r in parts})>1,
                answer_varies=len({r['answer'] for r in parts})>1,normalized_answer_varies=len({answer_norm(r['answer']) for r in parts})>1,
                contexts_by_batch={r['batch_size']:r['context_sha256'] for r in parts},answers_by_batch={r['batch_size']:r['answer'] for r in parts})
            motivation.append(value)
        for model,data in sorted({(r['model'],r['dataset']) for r in chosen}):
            sub=[r for r in motivation if (r['source'],r['model'],r['dataset'])==(source,model,data)]
            motivation_summary.append(dict(source=source,model=model,dataset=data,batch_sizes=sizes,n=len(sub),
                context_varies_questions=sum(r['context_varies'] for r in sub),answer_varies_questions=sum(r['answer_varies'] for r in sub),
                normalized_answer_varies_questions=sum(r['normalized_answer_varies'] for r in sub)))
    csvwrite(out/'batch_motivation_per_question.csv',motivation);csvwrite(out/'batch_motivation_summary.csv',motivation_summary)
    audit={'utc':utc(),'state':'complete','freeze_sha256':sha(OUT/'freeze.json'),'request_count':len(records),'counts':dict(counts),
        'quality_request_rows':len(exports),'random_context_rows':len(randomrows),'reader_runtime':load(OUT/'reader_runtime.json'),
        'gpu_occupancy':occupation[0],'stage1_status':'failed_86.49_percent','new_independent_test_samples':False,
        'same_stage2_samples':True,'method_sources_unchanged':True,'model_runs_added_in_analysis':False}
    save(out/'audit.json',audit);save(out/'manifest.json',{'utc':utc(),'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()}})
    print(__import__('json').dumps(audit,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
