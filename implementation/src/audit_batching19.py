import argparse
import collections
import fractions
import importlib.machinery
import importlib.util
import json
import math

import numpy as np

from common import ROOT, atomic_json, digest, read_jsonl
from cqc_budget12_core import file_sha, source_path, write_new
from batching19_core import check19, decision
from audit_cqc_budget12 import chat_text, interval, norm, packed, request_messages, tokens_for_rerank


from audit_cost17 import source_check as source_check17, tokenizers, latency_saving as original_latency_saving


def source_check(cfg,base):
    old=json.loads((ROOT/'configs/cost_stage17.json').read_text())
    previous,_,_=source_check17(old,ROOT/old['output'])
    rows=read_jsonl(base/'inputs.jsonl')
    assert rows==previous and len(rows)==len({r['id'] for r in rows})==32
    assert rows==read_jsonl(ROOT/cfg['candidate_source'])
    assert all(set(r)=={'id','question','documents','source_document_indices'} for r in rows)
    expected=[]
    for repeat in range(cfg['repetitions']):
        ordered=sorted(rows,key=lambda r:digest({'seed':cfg['seed'],'repeat':repeat,'id':r['id']}))
        for pos,row in enumerate(ordered):
            i=next(i for i,r in enumerate(rows) if r['id']==row['id'])
            arms=['full','batch4'] if (i+repeat)%2==0 else ['batch4','full']
            for slot,arm in enumerate(arms):expected.append({'repeat':repeat,'position':pos,'slot':slot,'id':row['id'],'arm':arm})
    schedule=read_jsonl(base/'schedule.jsonl');assert schedule==expected
    for row in rows:
        first=[s['arm'] for s in schedule if s['id']==row['id'] and s['slot']==0]
        assert first.count('full')==first.count('batch4')==1
    return rows,{},schedule


def preflight(cfg,base,rows,toks):
    maximum=cuts=bound=invocations=0
    for row in rows:
        lengths=[]
        for doc in row['documents']:
            ids,cut=tokens_for_rerank(toks['reranker'],row['question'],doc,cfg['reranker_max_length'])
            lengths.append(len(ids));maximum=max(maximum,len(ids));cuts+=cut
        invocations+=math.ceil(len(lengths)/cfg['batch_size'])
        prompt=chat_text(request_messages(row['question'],'',{},True))
        bound=max(bound,len(toks['reader'].encode(prompt,add_special_tokens=False))+cfg['context_tokens']+8)
    assert bound<=cfg['generator_max_input_tokens'] and toks['reranker'].pad_token_id is not None
    out={'state':'passed','manifest_sha256':file_sha(base/'manifest.json'),'questions':32,'max_reranker_tokens':maximum,'truncated_unique_original_inputs':cuts,'conservative_prompt_bound':bound,'batch4_forwards_per_pass':invocations,'primary_forward_invocations':cfg['repetitions']*(5631+invocations),'primary_scored_documents':22524,'labels_parsed':False,'gpu_model_loaded':False}
    atomic_json(base/'preflight.json',out);print(json.dumps(out),flush=True)


def verify_batches(r,row,cfg,toks):
    lengths=[v['input_tokens'] for v in r['reranker_inputs']]
    total=sum(lengths)
    assert r['nonpad_tokens']==total
    assert r['peak_allocated_bytes']>0 and r['peak_reserved_bytes']>=r['peak_allocated_bytes']
    if r['arm']=='full':
        assert r['microbatches'] is None and r['forward_invocations']==len(lengths) and r['padded_tokens']==total
        return
    ordered=sorted(enumerate(lengths),key=lambda v:(v[1],v[0]))
    groups=[[i for i,n in ordered[j:j+cfg['batch_size']]] for j in range(0,len(ordered),cfg['batch_size'])]
    assert [g['indices'] for g in r['microbatches']]==groups
    assert r['forward_invocations']==len(groups)
    assert r['padded_tokens']==sum(max(lengths[i] for i in g)*len(g) for g in groups)
    for record,group in zip(r['microbatches'],groups):
        width=max(lengths[i] for i in group);assert record['padded_width']==width
        assert set(record['tensor_sha256'])=={'input_ids','attention_mask','position_ids'}
        assert all(len(h)==64 for h in record['tensor_sha256'].values())
        if toks:
            tensors={k:[] for k in record['tensor_sha256']}
            for index in group:
                ids,cut=tokens_for_rerank(toks['reranker'],row['question'],row['documents'][index],cfg['reranker_max_length'])
                offset=width-len(ids)
                tensors['input_ids'].append([toks['reranker'].pad_token_id if j<offset else ids[j-offset] for j in range(width)])
                tensors['attention_mask'].append([int(j>=offset) for j in range(width)])
                tensors['position_ids'].append([max(0,j-offset) for j in range(width)])
            assert record['tensor_sha256']=={k:digest(v) for k,v in tensors.items()}


def audit(cfg, base, rows, ranks, schedule, toks):
    runtime = json.loads((base / 'runtime.json').read_text())
    assert runtime['state'] == 'complete' and runtime['manifest_sha256'] == file_sha(base / 'manifest.json')
    assert runtime['implementation_sha256'] == file_sha(ROOT / 'src/run_batching19.py')
    assert not runtime['answer_result_cache'] and not runtime['reranker_result_cache']
    for name, sha in runtime['output_sha256'].items():
        assert file_sha(base / name) == sha
    requests = read_jsonl(base / 'requests.jsonl')
    warmup = read_jsonl(base / 'warmup.jsonl')
    assert [{k: r[k] for k in ['repeat','position','slot','id','arm']} for r in requests] == schedule
    assert [(r['id'],r['arm']) for r in warmup] == [(r['id'], arm) for r in rows[:cfg['warmup_questions']] for arm in cfg['arms']]
    assert len(requests) == runtime['primary_requests'] == 128
    assert len(warmup) == runtime['warmup_requests'] == 4
    assert sum(len(r['scores']) for r in requests) == runtime['primary_score_calls'] == 22524
    assert sum(len(r['scores']) for r in warmup) == runtime['warmup_score_calls']
    by_id = {r['id']: r for r in rows}
    old_scores = {r['id']: r['views'][0]['scores'] for r in read_jsonl(ROOT / 'results/cqc_budget_stage12_pilot32/rankings.jsonl')}
    old_answers = {r['id']: r for r in read_jsonl(ROOT / 'results/cqc_budget_stage12_pilot32/answers.jsonl') if r['condition'] == 'original_neutral'}
    old_cache = {r['key']: r for r in read_jsonl(ROOT / 'results/cqc_budget_stage12_pilot32/answer_cache.jsonl')}
    encoded = {}
    max_prior_difference = 0.
    max_batched_prior_difference = 0.
    for r in warmup + requests:
        row = by_id[r['id']]
        indices = list(range(len(row['documents'])))
        verify_batches(r,row,cfg,toks)
        assert r['scored_indices'] == indices
        assert len(indices) == len(r['scores']) == len(r['reranker_inputs'])
        assert all(math.isfinite(v) for v in r['scores'])
        mapped = dict(zip(indices, r['scores']))
        assert r['order'] == sorted(indices, key=lambda i: (-mapped[i], i))
        for index, value, evidence in zip(indices, r['scores'], r['reranker_inputs']):
            if r['arm']=='full':
                max_prior_difference = max(max_prior_difference, abs(value-old_scores[r['id']][index]))
            else:
                max_batched_prior_difference = max(max_batched_prior_difference, abs(value-old_scores[r['id']][index]))
            assert 0 < evidence['input_tokens'] <= cfg['reranker_max_length']
            if toks:
                key = (r['id'], index)
                if key not in encoded:
                    ids, cut = tokens_for_rerank(toks['reranker'], row['question'], row['documents'][index], cfg['reranker_max_length'])
                    encoded[key] = {'input_ids_sha256': digest(ids), 'input_tokens': len(ids), 'truncated': cut}
                assert evidence == encoded[key]
        p = r['packing']
        assert len(p['selected']) == len(set(p['selected'])) and set(p['selected']) <= set(r['order'][:cfg['top_k']])
        assert p['selected'] == [i for i in r['order'][:cfg['top_k']] if i in p['selected']]
        assert p['context'] == '\n\n'.join('Title: ' + row['documents'][i]['title'] + '\n' + row['documents'][i]['text'] for i in p['selected'])
        assert 0 <= p['context_tokens'] <= cfg['context_tokens']
        if toks:
            assert p == packed(row['documents'], r['order'], toks['reader'], cfg['context_tokens'], cfg['top_k'])
        messages = request_messages(row['question'], p['context'], {}, True)
        assert r['prompt'] == chat_text(messages)
        assert r['options'] == {'do_sample': False, 'max_new_tokens': cfg['max_new_tokens'], 'pad_token_id': runtime['reader_eos_token_id'], 'repetition_penalty': 1.0}
        assert 0 < len(r['output_ids']) <= cfg['max_new_tokens']
        assert r['cap_hit'] == (len(r['output_ids']) == cfg['max_new_tokens'])
        assert 0 < r['input_tokens'] <= cfg['generator_max_input_tokens']
        if toks:
            ids = toks['reader'].encode(r['prompt'], add_special_tokens=False)
            assert len(ids) == r['input_tokens'] and digest(ids) == r['input_ids_sha256']
            assert toks['reader'].decode(r['output_ids'], skip_special_tokens=True).strip() == r['answer']
        seconds = r['seconds']
        assert set(seconds) == {'selection','reranker_encode','reranker_forward','ordering_pack','reader_prompt','reader_generate','reader_decode','total'}
        assert all(math.isfinite(v) and v >= 0 for v in seconds.values()) and seconds['total'] > 0
        assert math.isclose(seconds['total'], sum(v for k,v in seconds.items() if k != 'total'), rel_tol=1e-10, abs_tol=1e-8)
        assert set(r['gpu_before']['pids']) <= {runtime['pid']} and set(r['gpu_after']['pids']) <= {runtime['pid']}
    grouped = collections.defaultdict(list)
    for r in requests:
        grouped[(r['id'],r['arm'])].append(r)
    repeats, full_changes = [], []
    max_repeat_difference = 0.
    for (qid, arm), values in grouped.items():
        assert [r['repeat'] for r in values] == list(range(cfg['repetitions']))
        for other in values[1:]:
            difference = max(abs(a-b) for a,b in zip(values[0]['scores'],other['scores']))
            max_repeat_difference = max(max_repeat_difference, difference)
            if other['output_ids'] != values[0]['output_ids'] or other['packing'] != values[0]['packing'] or other['reranker_inputs'] != values[0]['reranker_inputs']:
                repeats.append({'id': qid, 'arm': arm, 'repeat': other['repeat']})
        if arm == 'full':
            previous = old_cache[old_answers[qid]['key']]
            for r in values:
                if previous['output_ids'] != r['output_ids'] or previous['prompt'] != r['prompt']:
                    full_changes.append({'id': qid, 'repeat': r['repeat']})
    engineering = not repeats and not full_changes and max_repeat_difference <= cfg['reranker_repeat_atol'] and max_prior_difference <= cfg['reranker_repeat_atol']
    details = {'state': 'passed', 'tokenization_reconstructed': bool(toks), 'engineering_gate_passed': engineering, 'primary_requests': len(requests), 'warmup_requests': len(warmup), 'primary_score_calls': runtime['primary_score_calls'], 'warmup_score_calls': runtime['warmup_score_calls'], 'max_stage12_original_score_difference': max_prior_difference, 'max_repeat_score_difference': max_repeat_difference, 'max_batch4_prior_score_difference': max_batched_prior_difference, 'cross_batch_scores_within_1e4': max_batched_prior_difference <= cfg['reranker_repeat_atol'], 'repeat_mismatches': repeats, 'full_baseline_mismatches': full_changes, 'labels_parsed_during_structural_audit': False, 'manifest_sha256': file_sha(base / 'manifest.json')}
    return requests, grouped, runtime, details


def evaluate(cfg,base,rows,requests,grouped,runtime,details):
    loader=importlib.machinery.SourceFileLoader('hotpot_batching19',str(ROOT/'references/hotpot_eval.txt'))
    evaluator=importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name,loader));loader.exec_module(evaluator)
    labels={r['id']:r for r in read_jsonl(ROOT/'data/stage08_confirmation/labels.jsonl')}
    pairs=[];differences=[]
    for row in rows:
        label=labels[row['id']];assert label['question']==row['question']
        first={arm:grouped[(row['id'],arm)][0] for arm in cfg['arms']}
        metrics={};medians={}
        for arm in cfg['arms']:
            answer=first[arm]['answer']
            metrics[arm]={'answer':answer,'f1':float(evaluator.f1_score(answer,label['answer'])[0]),'em':float(evaluator.exact_match_score(answer,label['answer']))}
            medians[arm]=float(np.median([r['seconds']['total'] for r in grouped[(row['id'],arm)]]))
        drift=[abs(a-b) for a,b in zip(first['full']['scores'],first['batch4']['scores'])];differences.extend(drift)
        pairs.append({'id':row['id'],'metrics':metrics,'latency_medians_seconds':medians,'context_changed':first['full']['packing']['context']!=first['batch4']['packing']['context'],'output_tokens_changed':first['full']['output_ids']!=first['batch4']['output_ids'],'normalized_answer_changed':norm(first['full']['answer'])!=norm(first['batch4']['answer']),'full_ranking_changed':first['full']['order']!=first['batch4']['order'],'top10_ranking_changed':first['full']['order'][:10]!=first['batch4']['order'][:10],'max_score_difference':max(drift),'scores_over_1e4':sum(d>cfg['reranker_repeat_atol'] for d in drift)})
    means={arm:{m:sum(r['metrics'][arm][m] for r in pairs)/len(pairs) for m in ['f1','em']} for arm in cfg['arms']}
    paired={m:interval([r['metrics']['batch4'][m]-r['metrics']['full'][m] for r in pairs],cfg['seed'],cfg['bootstrap_replicates']) for m in ['f1','em']}
    cost=original_latency_saving([r['latency_medians_seconds']['full'] for r in pairs],[r['latency_medians_seconds']['batch4'] for r in pairs],cfg['seed'],cfg['bootstrap_replicates'])
    cost['batched_sum_question_medians_seconds']=cost.pop('shortlist_sum_question_medians_seconds')
    counts={arm:sum(len(r['scores']) for r in requests if r['arm']==arm) for arm in cfg['arms']}
    invocations={arm:sum(r['forward_invocations'] for r in requests if r['arm']==arm) for arm in cfg['arms']}
    diagnostics={}
    for arm in cfg['arms']:
        vals=[r for r in requests if r['arm']==arm];times=[r['seconds']['total'] for r in vals]
        diagnostics[arm]={'requests':len(vals),'median_seconds':float(np.median(times)),'p95_seconds':float(np.quantile(times,.95)),'component_totals_seconds':{k:sum(v['seconds'][k] for v in vals) for k in vals[0]['seconds']},'repeat_totals_seconds':[sum(v['seconds']['total'] for v in vals if v['repeat']==rep) for rep in range(cfg['repetitions'])],'padded_tokens':sum(v['padded_tokens'] for v in vals),'nonpad_tokens':sum(v['nonpad_tokens'] for v in vals),'peak_allocated_bytes':max(v['peak_allocated_bytes'] for v in vals),'peak_reserved_bytes':max(v['peak_reserved_bytes'] for v in vals),'cap_hits_first_repeat':sum(v['cap_hit'] for v in vals if v['repeat']==0)}
    contexts=sum(r['context_changed'] for r in pairs);tokens=sum(r['output_tokens_changed'] for r in pairs)
    gate=decision(cfg,details['engineering_gate_passed'],contexts,tokens,paired,cost)
    out={'methods':means,'primary_contrast':'batch4_minus_full','paired':paired,'latency_saving':cost,'primary_scored_documents':counts,'scored_document_saving':1-counts['batch4']/counts['full'],'primary_forward_invocations':invocations,'forward_invocation_saving':1-invocations['batch4']/invocations['full'],'primary_generation_calls':len(requests),'warmup_generation_calls':runtime['warmup_requests'],'warmup_scored_documents':runtime['warmup_score_calls'],'context_changes':contexts,'output_token_changes':tokens,'normalized_answer_changes':sum(r['normalized_answer_changed'] for r in pairs),'full_ranking_changes':sum(r['full_ranking_changed'] for r in pairs),'top10_ranking_changes':sum(r['top10_ranking_changed'] for r in pairs),'cross_batch_score_drift':{'unique_documents':len(differences),'max':max(differences),'mean':float(np.mean(differences)),'p95':float(np.quantile(differences,.95)),'above_1e4':sum(v>cfg['reranker_repeat_atol'] for v in differences),'within_1e4':max(differences)<=cfg['reranker_repeat_atol'],'diagnostic_not_operational_gate':True},'decision':gate,'diagnostics':diagnostics,'scope':cfg['scope'],'bootstrap':{'replicates':cfg['bootstrap_replicates'],'seed':cfg['seed'],'unit':'question','repeated_timings_are_not_extra_questions':True,'multiplicity_adjusted':False},'manifest_sha256':file_sha(base/'manifest.json'),'labels_sha256':file_sha(ROOT/'data/stage08_confirmation/labels.jsonl'),'official_evaluator_sha256':file_sha(ROOT/'references/hotpot_eval.txt')}
    return pairs,out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','full','replay']);args=p.parse_args()
    cfg,base,_=check19();rows,ranks,schedule=source_check(cfg,base)
    toks=tokenizers(cfg) if args.mode!='replay' else {}
    if args.mode=='preflight':preflight(cfg,base,rows,toks)
    else:
        target=base/('analysis_remote' if args.mode=='full' else 'analysis_local');assert not target.exists()
        requests,grouped,runtime,details=audit(cfg,base,rows,ranks,schedule,toks)
        pairs,summary=evaluate(cfg,base,rows,requests,grouped,runtime,details)
        target.mkdir();write_new(target/'question_pairs.jsonl',pairs);atomic_json(target/'analysis.json',summary)
        atomic_json(target/'audit.json',{**details,'analysis_sha256':file_sha(target/'analysis.json'),'implementation_sha256':file_sha(__file__),'result_sha256':{v.name:file_sha(v) for v in base.iterdir() if v.is_file() and not v.name.startswith('queue_')}})
        print(json.dumps({'state':'audit_complete','engineering':details,'methods':summary['methods'],'paired':summary['paired'],'latency_saving':summary['latency_saving'],'decision':summary['decision']}),flush=True)
