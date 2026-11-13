"""Independent CPU audit and complete numerical exports; never selects evaluation hyperparameters."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections
import csv
import importlib.util
import json
import math
import statistics
from eswa_stage2_common import *
from eswa_stage2_metrics import *


def mean(values):
    values=[v for v in values if v is not None]
    return sum(values)/len(values) if values else None


def csvwrite(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})


def ident(r):return r['model'],r['id'],r['method'],r['batch_size'],r['rep']


def independent_pack(row,scores,tok):
    ranking=sorted(range(len(scores)),key=lambda i:(-scores[i],i));selected=[];text=''
    for i in ranking[:10]:
        d=row['documents'][i];block='Title: '+d['title']+'\n'+d['text']
        proposal=text+'\n\n'+block if selected else block
        if len(tok.encode(proposal,add_special_tokens=False))<=512:selected.append(i);text=proposal
    return ranking,selected,text,len(tok.encode(text,add_special_tokens=False))


def main():
    check()
    from transformers import AutoTokenizer
    from run_eswa_stage2_v1 import verify_freeze,schedule,Packer
    from eswa_gate_methods_v2 import flagged,order
    freeze=verify_freeze();runtime=load(BASE/'reader_runtime.json');assert runtime['state']=='complete'
    assert sha(BASE/'answers.jsonl')==runtime['answers_sha256']
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen--Qwen3-8B'),local_files_only=True)
    spec=importlib.util.spec_from_file_location('official_qasper',DATA/'qasper_evaluator.py');official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    raw=load(DATA/'qasper-test-v0.3.json');refs=official.get_answers_and_evidence(raw,text_evidence_only=True)
    rows={r['id']:r for r in read_jsonl(BASE/'inputs.jsonl')};hot={r['_id']:r for r in read_jsonl(DATA/'hotpot_labels.jsonl')}
    answers={r['key']:r for r in read_jsonl(BASE/'answers.jsonl')};link_rows=read_jsonl(BASE/'reader_links.jsonl')
    links={ident(r):r['key'] for r in link_rows};assert len(links)==len(link_rows)
    records={};all_records=[]
    for kind in ['qwen','bge']:
        rr=read_jsonl(BASE/f'requests_{kind}.jsonl');info=load(BASE/f'runtime_{kind}.json')
        assert info['state']=='complete' and info['parameters_unchanged'] and info['buffers_unchanged']
        assert sha(BASE/f'requests_{kind}.jsonl')==info['records_sha256']
        expected={(kind,r['id'],r['method'],r['batch_size'],r['rep']) for r in schedule(kind)}
        assert {ident(r) for r in rr}==expected and len(rr)==len(expected)
        all_records+=rr;records.update({ident(r):r for r in rr})
    assert len(records)==len(all_records)==len(links)==10944 and set(records)==set(links)
    assert set(answers)=={r['key'] for r in read_jsonl(BASE/'reader_prompts.jsonl')}
    encoded={};replay_cache={};checks=collections.Counter();max_rescore_error=0.;exports=[];change_contexts=[]
    recovery={};recovery_runtime=None;recovery_exports=[]
    if (BASE/'timing_recovery_plan.json').exists():
        plan=load(BASE/'timing_recovery_plan.json');recovery_runtime=load(BASE/'timing_recovery_runtime.json')
        assert recovery_runtime['state']=='complete' and recovery_runtime['parameters_unchanged'] and recovery_runtime['buffers_unchanged']
        assert sha(BASE/'timing_recovery_qwen.jsonl')==recovery_runtime['records_sha256']
        recovered=read_jsonl(BASE/'timing_recovery_qwen.jsonl');recovery={ident(r):r for r in recovered}
        assert len(recovery)==len(recovered)==len(plan['requests'])
        assert set(recovery)=={tuple(r['key']) for r in plan['requests']}
        for entry in plan['requests']:assert digest(records[tuple(entry['key'])])==entry['original_record_digest']
        for r in recovered:
            previous=records[ident(r)];assert previous['run_id']==plan['invalid_run_id'] and r['post_request_gpu_check']
            assert independent_pack(rows[r['id']],r['scores'],tok)==(r['order'],r['selected'],r['context'],r['tokens'])
            recovery_exports.append({'model':r['model'],'id':r['id'],'dataset':r['dataset'],'method':r['method'],'batch_size':r['batch_size'],'rep':r['rep'],
                'original_run_id':r['original_run_id'],'original_total_seconds':previous['total_seconds'],'recovered_total_seconds':r['total_seconds'],
                'original_selected':previous['selected'],'recovered_selected':r['selected'],'context_same_as_original':r['context']==previous['context'],
                'scores_same_as_original':r['scores']==previous['scores'],'changed_fields':r['changed_fields'],
                'original_rescored_count':previous['rescored_count'],'recovered_rescored_count':r['rescored_count'],
                'original_context_sha256':digest(previous['context']),'recovered_context_sha256':digest(r['context']),
                'use':'timing recovery only; scheduled rep0/rep1 remain primary quality data'})
    effective={ident(r):recovery.get(ident(r),r) for r in all_records}
    medians={};raw_medians={}
    for r in all_records:
        key_=(r['model'],r['id'],r['method'],r['batch_size'])
        if key_ not in medians:
            medians[key_]=statistics.median(effective[key_+(rep,)]['total_seconds'] for rep in range(2))
            raw_medians[key_]=statistics.median(records[key_+(rep,)]['total_seconds'] for rep in range(2))
    for r in all_records:
        row=rows[r['id']];qkey=(r['model'],r['id']);encoded.setdefault(qkey,r['encoded_sha256']);assert encoded[qkey]==r['encoded_sha256']
        assert len(r['scores'])==len(row['documents'])==r['candidate_count']
        pk=(row['id'],tuple(r['scores']))
        if pk not in replay_cache:replay_cache[pk]=independent_pack(row,r['scores'],tok)
        ranking,selected,text,tokens=replay_cache[pk]
        assert ranking==r['order'] and selected==r['selected'] and text==r['context'] and tokens==r['tokens']
        checks['independent_packs']+=1
        reference=records[r['model'],r['id'],'serial',1,r['rep']]
        batch=records[r['model'],r['id'],'batch4',4,r['rep']]
        serial_other=records[r['model'],r['id'],'serial',1,1-r['rep']]
        assert r['total_seconds']>0 and r['forward_seconds']>=0
        work=list(r['initial_scores']);marked=set();packer=Packer(row['documents'],tok)
        if r['method'] in ['gate_v1','gate_v2','top15']:
            checks['initial_scores_differ_from_batch4']+=int(work!=batch['scores'])
            for iteration,round_ in enumerate(r['rounds']):
                indices=round_['indices'];assert not marked&set(indices)
                if r['method']=='top15':assert iteration==0 and indices==order(work)[:15]
                else:
                    fresh,pairs=flagged(work,marked,packer,r['tau'],r['method'][-2:]);assert fresh==indices and pairs==round_['trigger_pairs']
                values={int(i):s for i,s in round_['actual_serial_scores'].items()};assert set(indices)==set(values)
                for i,s in values.items():
                    error=abs(s-reference['scores'][i]);max_rescore_error=max(max_rescore_error,error)
                    checks['actual_serial_values_checked']+=1;checks['actual_serial_values_different']+=int(error!=0)
                    work[i]=s
                marked.update(indices)
            if r['method'].startswith('gate_'):assert flagged(work,marked,packer,r['tau'],r['method'][-2:])[0]==[]
        assert work==r['scores'] and sorted(marked)==r['rescored_indices'] and len(marked)==r['rescored_count']
        assert r['rescored_fraction']==len(marked)/len(work)
        pred=answers[links[ident(r)]];documents=[row['documents'][i] for i in selected]
        if row['dataset']=='qasper':
            metric=qasper_evidence(documents,refs[row['id']],official)
            metric.update(answer_f1=max(official.token_f1_score(pred['answer'],a['answer']) for a in refs[row['id']]),
                          answer_em=max(float(official.normalize_answer(pred['answer'])==official.normalize_answer(a['answer'])) for a in refs[row['id']]))
        else:metric={**hotpot_evidence(documents,hot[row['id']]),**hotpot_answer(pred['answer'],hot[row['id']]['answer'])}
        other=records[r['model'],r['id'],r['method'],r['batch_size'],1-r['rep']]
        key_=(r['model'],r['id'],r['method'],r['batch_size']);serial_time=medians[r['model'],r['id'],'serial',1]
        export={k:r[k] for k in ['model','id','dataset','paper_id','method','batch_size','rep','role','tau','candidate_count','selected','rescored_indices','rescored_count','rescored_fraction','total_seconds','forward_seconds','gate_check_seconds','forward_calls','pack_requests','pack_cache_misses']}
        export.update(serial_selected=reference['selected'],context_equal_serial=text==reference['context'],members_equal_serial=set(selected)==set(reference['selected']),
                      context_sha256=digest(text),serial_context_sha256=digest(reference['context']),context_tokens=tokens,
                      repeat_context_identical=text==other['context'],repeat_scores_identical=r['scores']==other['scores'],
                      two_repeat_median_seconds=medians[key_],serial_two_repeat_median_seconds=serial_time,paired_time_ratio_to_serial=medians[key_]/serial_time,
                      actual_rescoring_values_match_serial=all(s==reference['scores'][int(i)] for rd in r['rounds'] for i,s in rd['actual_serial_scores'].items()),
                      truncated_candidates=sum(r['truncated']),reader_key=pred['key'],answer=pred['answer'],reader_cap_hit=pred['cap_hit'],
                      raw_timing_isolation_certain=ident(r) not in recovery,timing_source='separate_recovery' if ident(r) in recovery else 'scheduled',
                      effective_total_seconds=effective[ident(r)]['total_seconds'],effective_forward_seconds=effective[ident(r)]['forward_seconds'],
                      effective_gate_check_seconds=effective[ident(r)]['gate_check_seconds'],
                      raw_two_repeat_median_seconds=raw_medians[key_],raw_serial_two_repeat_median_seconds=raw_medians[r['model'],r['id'],'serial',1],
                      error='original_timing_isolation_uncertain_recovered' if ident(r) in recovery else '',**metric)
        exports.append(export)
        if r['rep']==0 and text!=reference['context']:change_contexts.append({k:export[k] for k in ['model','dataset','id','method','batch_size','selected','serial_selected','context_sha256','serial_context_sha256']})
    metadata=[]
    for row in rows.values():
        if row['role']!='evaluation':continue
        if row['dataset']=='qasper':
            metric=qasper_evidence(row['documents'],refs[row['id']],official)
            gold={ws(t) for a in refs[row['id']] for t in a['evidence']};paper={ws(p) for s in raw[row['paper_id']]['full_text'] for p in s['paragraphs']}
            extra={'gold_paragraphs':len(gold),'gold_unmapped_to_paper':len(gold-paper),
                   'unanswerable_reference':any(a['type']=='none' for a in refs[row['id']]),'only_unanswerable':all(a['type']=='none' for a in refs[row['id']])}
        else:metric=hotpot_evidence(row['documents'],hot[row['id']]);extra={'gold_paragraphs':None,'gold_unmapped_to_paper':None}
        metadata.append({'id':row['id'],'dataset':row['dataset'],'paper_id':row.get('paper_id',''),'candidates':len(row['documents']),**metric,**extra})
    by_group=collections.defaultdict(list)
    for r in exports:by_group[r['model'],r['dataset'],r['method'],r['batch_size'],r['rep']].append(r)
    summary=[]
    for (model,dataset,method,size,rep),rr in sorted(by_group.items()):
        base=by_group[model,dataset,'batch4',4,rep];base_diff=sum(not r['context_equal_serial'] for r in base);diff=sum(not r['context_equal_serial'] for r in rr)
        fraction=mean([r['rescored_fraction'] for r in rr]);reduction=1-diff/base_diff if base_diff else None
        zero_upper=1-.05**(1/len(rr)) if diff==0 else None
        summary.append({'model':model,'dataset':dataset,'method':method,'batch_size':size,'rep':rep,'n':len(rr),
                        'context_differences':diff,'member_differences':sum(not r['members_equal_serial'] for r in rr),
                        'baseline_context_differences':base_diff,'context_difference_reduction':reduction,
                        'mean_rescored_fraction':fraction,'max_rescored_fraction':max(r['rescored_fraction'] for r in rr),
                        'joint_90pct_20pct':bool(reduction is not None and reduction>=.9-1e-12 and fraction<=.2),
                        'strictly_over_90pct_and_within20pct':bool(reduction is not None and reduction>.9+1e-12 and fraction<=.2),
                        'baseline_zero_unidentified':base_diff==0,'zero_change_one_sided95_upper':zero_upper,
                        'gold_recall':mean([r['gold_evidence_recall'] for r in rr]),'gold_recall_n':sum(r['gold_evidence_recall'] is not None for r in rr),
                        'gold_metric':rr[0]['gold_metric'],'support_title_recall':mean([r['support_title_recall'] for r in rr]),
                        'answer_f1':mean([r['answer_f1'] for r in rr]),'answer_em':mean([r['answer_em'] for r in rr]),
                        'sum_query_median_seconds':sum(r['two_repeat_median_seconds'] for r in rr),'mean_query_median_seconds':mean([r['two_repeat_median_seconds'] for r in rr]),
                        'aggregate_time_ratio_to_serial':sum(r['two_repeat_median_seconds'] for r in rr)/sum(r['serial_two_repeat_median_seconds'] for r in rr),
                        'raw_sum_query_median_seconds':sum(r['raw_two_repeat_median_seconds'] for r in rr),
                        'raw_aggregate_time_ratio_to_serial':sum(r['raw_two_repeat_median_seconds'] for r in rr)/sum(r['raw_serial_two_repeat_median_seconds'] for r in rr),
                        'raw_timing_uncertain_records':sum(not r['raw_timing_isolation_certain'] for r in rr),
                        'mean_paired_time_ratio_to_serial':mean([r['paired_time_ratio_to_serial'] for r in rr]),
                        'repeat_context_changes':sum(not r['repeat_context_identical'] for r in rr),'repeat_score_changes':sum(not r['repeat_scores_identical'] for r in rr),
                        'reader_cap_hits':sum(r['reader_cap_hit'] for r in rr),'truncated_candidates':sum(r['truncated_candidates'] for r in rr)})
    official_checks=[]
    for key_,rr in by_group.items():
        if key_[1]!='qasper':continue
        predicted={r['id']:{'answer':r['answer'],'evidence':list(dict.fromkeys(rows[r['id']]['documents'][i]['text'] for i in r['selected']))} for r in rr}
        result=official.evaluate({qid:refs[qid] for qid in predicted},predicted)
        assert abs(result['Answer F1']-mean([r['answer_f1'] for r in rr]))<1e-12
        assert abs(result['Evidence F1']-mean([r['official_evidence_f1'] for r in rr]))<1e-12
        official_checks.append({'group':key_,'Answer F1':result['Answer F1'],'Evidence F1':result['Evidence F1']})
    # Preserve every BGE calibration tau at question level, using saved calibration scores only.
    cal=read_jsonl(BASE/'bge_calibration_requests.jsonl');cal={(r['id'],r['method']):r for r in cal};cal_exports=[]
    for row in rows.values():
        if row['role']!='bge_calibration':continue
        a=cal[row['id'],'serial'];b=cal[row['id'],'batch4']
        for tau in CFG['tau_grid']:
            values=list(b['scores']);marked=set();packer=Packer(row['documents'],tok)
            while True:
                fresh,_=flagged(values,marked,packer,tau,'v1')
                if not fresh:break
                for i in fresh:values[i]=a['scores'][i]
                marked.update(fresh)
            packed=packer(order(values))
            cal_exports.append({'id':row['id'],'paper_id':row['paper_id'],'tau':tau,'selected':packed['selected'],'serial_selected':a['selected'],
                                'context_equal_serial':packed['context']==a['context'],'members_equal_serial':set(packed['selected'])==set(a['selected']),
                                'rescored_count':len(marked),'rescored_fraction':len(marked)/len(values),'mode':'saved_calibration_score_simulation'})
    selected_tau=load(BASE/'bge_tau_selection.json')
    for cell in selected_tau['grid']:
        rr=[r for r in cal_exports if r['tau']==cell['tau']]
        assert sum(not r['context_equal_serial'] for r in rr)==cell['context_differences']
        assert abs(mean([r['rescored_fraction'] for r in rr])-cell['mean_rescored_fraction'])<1e-12
    report={'utc':utc(),'state':'complete','stage0_status':'passed','stage1_status':'failed_86.49_percent_not_90',
            'continuation':'explicit_user_decision','stage2_executed_complete':True,'write_paper':False,
            'audit_counts':dict(checks),'maximum_actual_serial_score_error':max_rescore_error,'unique_replayed_score_vectors':len(replay_cache),
            'freeze_sha256':sha(BASE/'freeze.json'),'bge_tau':selected_tau['tau'],'reader':runtime,'official_qasper_checks':official_checks,
            'historical_128_role':'development','interruption_files':[p.name for p in BASE.glob('interruption*')],
            'metric_definitions_sha256':sha(ROOT/'src/eswa_stage2_metrics.py'),
            'timing_recovery':recovery_runtime,'timing_recovered_records':len(recovery),
            'timing_recovery_protocol_deviation':bool(recovery),
            'quality_statistics_use_scheduled_records_only':True}
    out=BASE/'analysis';out.mkdir(exist_ok=False)
    csvwrite(out/'per_question_repeats.csv',exports)
    csvwrite(out/'per_question_medians.csv',[r for r in exports if r['rep']==0])
    csvwrite(out/'all_results.csv',summary)
    csvwrite(out/'main_results.csv',[r for r in summary if r['rep']==0 and r['method']!='batch_scan'])
    csvwrite(out/'qwen_qasper_batch_scan.csv',[r for r in summary if r['model']=='qwen' and r['dataset']=='qasper' and r['method'] in ['serial','batch4','batch_scan']])
    csvwrite(out/'candidate_gold_audit.csv',metadata)
    csvwrite(out/'changed_contexts_index.csv',change_contexts)
    csvwrite(out/'bge_calibration_per_question_all_tau.csv',cal_exports)
    csvwrite(out/'bge_calibration_grid.csv',selected_tau['grid'])
    if recovery_exports:csvwrite(out/'timing_recovery_per_request.csv',recovery_exports)
    save(out/'audit.json',report)
    save(out/'export_manifest.json',{'utc':utc(),'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()}})
    print(json.dumps({'state':'analysis_complete','records':len(exports),'summary_cells':len(summary),'audits':dict(checks),'output':str(out)}),flush=True)


if __name__=='__main__':main()
