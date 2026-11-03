import argparse
import collections
import importlib.machinery
import importlib.util
import itertools
import json
import math

import numpy as np

from common import ROOT, atomic_json, digest, read_jsonl
from cqc_budget12_core import file_sha, source_path, write_new
from ipm_crossset_v3_core import check22, decision, schedule as frozen_schedule, stability, wilson
from audit_cqc_budget12 import chat_text, interval, norm, packed, request_messages, tokens_for_rerank
from audit_cost18 import tokenizers, latency_saving
from audit_batching19 import verify_batches


def source_check(cfg,base):
    rows=read_jsonl(base/'inputs.jsonl');queries=read_jsonl(ROOT/cfg['data']/'queries.jsonl')
    assert [{'id':r['id'],'question':r['question']} for r in rows]==queries and len(rows)==cfg['questions']==96
    split=json.loads((ROOT/cfg['data']/'split.json').read_text())
    assert [r['id'] for r in rows]==split['ids']
    assert file_sha(ROOT/cfg['data']/'queries.jsonl')==split['queries_sha256']
    assert not set(split['selected_normalized_question_hashes'])&set(split['prior_normalized_question_hashes'])
    seq=read_jsonl(base/'schedule.jsonl');assert seq==frozen_schedule(rows,cfg)
    retrieval=json.loads((base/'retrieval_audit.json').read_text());assert retrieval['state']=='passed'
    return rows,{},seq


def preflight(cfg,base,rows,toks):
    maximum=cuts=bound=0
    for row in rows:
        for doc in row['documents']:
            ids,cut=tokens_for_rerank(toks['reranker'],row['question'],doc,cfg['reranker_max_length'])
            maximum=max(maximum,len(ids));cuts+=cut
        prompt=chat_text(request_messages(row['question'],'',{},True))
        bound=max(bound,len(toks['reader'].encode(prompt,add_special_tokens=False))+cfg['context_tokens']+8)
    assert bound<=cfg['generator_max_input_tokens'] and toks['reranker'].pad_token_id is not None
    out={'state':'passed','manifest_sha256':file_sha(base/'manifest.json'),'questions':len(rows),'max_reranker_tokens':maximum,'truncated_unique_original_inputs':cuts,'conservative_prompt_bound':bound,'primary_requests':len(rows)*len(cfg['arms'])*cfg['repetitions'],'primary_scored_documents':sum(len(r['documents']) for r in rows)*len(cfg['arms'])*cfg['repetitions'],'labels_parsed':False,'gpu_model_loaded':False}
    atomic_json(base/'preflight.json',out);print(json.dumps(out),flush=True)


def verify_layout(r,row,cfg,toks,runtime):
    copied=dict(r)
    if r['arm']=='bf16_serial':copied['arm']='full'
    verify_batches(copied,row,{**cfg,'batch_size':1 if r['arm']=='fp32_serial' else 4},toks)
    native=runtime['native_backend_flags']
    assert r['reader_native_backend_flags']==r['backend_flags_after']==native
    expected=native if r['arm'].startswith('bf16_') else {'math':True,'flash':False,'efficient':False,'cudnn':False}
    assert r['reranker_backend_flags']==expected
    assert r['parameter_dtype']==('torch.bfloat16' if r['arm'].startswith('bf16_') else 'torch.float32')
    assert math.isfinite(r['dtype_transition_seconds_excluded']) and r['dtype_transition_seconds_excluded']>=0


def audit(cfg, base, rows, ranks, schedule, toks):
    runtime = json.loads((base / 'runtime.json').read_text())
    assert runtime['state'] == 'complete' and runtime['manifest_sha256'] == file_sha(base / 'manifest.json')
    assert runtime['implementation_sha256'] == file_sha(ROOT / 'src/run_ipm_crossset_v3.py')
    assert not runtime['answer_result_cache'] and not runtime['reranker_result_cache']
    for name, sha in runtime['output_sha256'].items():
        assert file_sha(base / name) == sha
    requests = read_jsonl(base / 'requests.jsonl')
    warmup = read_jsonl(base / 'warmup.jsonl')
    assert [{k: r[k] for k in ['repeat','position','slot','id','arm']} for r in requests] == schedule
    assert [(r['id'],r['arm']) for r in warmup] == [(r['id'], arm) for r in rows[:cfg['warmup_questions']] for arm in cfg['arms']]
    assert len(requests) == runtime['primary_requests'] == len(rows)*len(cfg['arms'])*cfg['repetitions']
    assert len(warmup) == runtime['warmup_requests'] == cfg['warmup_questions']*len(cfg['arms'])
    assert sum(len(r['scores']) for r in requests) == runtime['primary_score_calls'] == sum(len(r['documents']) for r in rows)*len(cfg['arms'])*cfg['repetitions']
    assert sum(len(r['scores']) for r in warmup) == runtime['warmup_score_calls']
    by_id = {r['id']: r for r in rows}
    encoded = {}
    for r in warmup + requests:
        row = by_id[r['id']]
        indices = list(range(len(row['documents'])))
        verify_layout(r,row,cfg,toks,runtime)
        assert r['scored_indices'] == indices
        assert len(indices) == len(r['scores']) == len(r['reranker_inputs'])
        assert all(math.isfinite(v) for v in r['scores'])
        mapped = dict(zip(indices, r['scores']))
        assert r['order'] == sorted(indices, key=lambda i: (-mapped[i], i))
        for index, value, evidence in zip(indices, r['scores'], r['reranker_inputs']):
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
    assert runtime['weights_before']==runtime['weights_after'] and runtime['buffers_before']==runtime['buffers_after']
    assert not runtime['encoding_cache'] and runtime['torch_version']==cfg['expected_torch'] and runtime['transformers_version']==cfg['expected_transformers']
    engineering = not repeats and max_repeat_difference <= cfg['reranker_repeat_atol']
    details = {'state': 'passed', 'tokenization_reconstructed': bool(toks), 'engineering_gate_passed': engineering, 'primary_requests': len(requests), 'warmup_requests': len(warmup), 'primary_score_calls': runtime['primary_score_calls'], 'warmup_score_calls': runtime['warmup_score_calls'], 'prior_score_reproduction_applicable':False, 'max_repeat_score_difference': max_repeat_difference, 'repeat_mismatches': repeats, 'full_baseline_mismatches': full_changes, 'labels_parsed_during_structural_audit': False, 'manifest_sha256': file_sha(base / 'manifest.json')}
    return requests, grouped, runtime, details


def evaluate(cfg,base,rows,requests,grouped,runtime,details):
    from musique_metrics import score_answer
    labels={r['id']:r for r in read_jsonl(ROOT/cfg['labels'])}
    contrasts=[('bf16_batch4_minus_bf16_serial','bf16_serial','bf16_batch4'),('fp32_batch4_minus_fp32_serial','fp32_serial','fp32_batch4'),('fp32_batch4_minus_bf16_serial','bf16_serial','fp32_batch4'),('fp32_serial_minus_bf16_serial','bf16_serial','fp32_serial')]
    pairs=[];drifts={name:[] for name,_,_ in contrasts}
    for row in rows:
        label=labels[row['id']];assert label['question']==row['question']
        first={arm:grouped[(row['id'],arm)][0] for arm in cfg['arms']}
        metrics={arm:{'answer':r['answer'],**score_answer(r['answer'],label['answer'],label['answer_aliases'])} for arm,r in first.items()}
        times={arm:float(np.median([r['seconds']['total'] for r in grouped[(row['id'],arm)]])) for arm in cfg['arms']}
        compare={}
        for name,a,b in contrasts:
            x,y=first[a],first[b];delta=[abs(v-w) for v,w in zip(x['scores'],y['scores'])];drifts[name].extend(delta)
            compare[name]={'context_changed':x['packing']['context']!=y['packing']['context'],'membership_changed':set(x['packing']['selected'])!=set(y['packing']['selected']),'output_tokens_changed':x['output_ids']!=y['output_ids'],'normalized_answer_changed':norm(x['answer'])!=norm(y['answer']),'full_ranking_changed':x['order']!=y['order'],'top10_ranking_changed':x['order'][:10]!=y['order'][:10],'max_score_difference':max(delta),'scores_over_1e4':sum(v>1e-4 for v in delta)}
        pairs.append({'id':row['id'],'metrics':metrics,'latency_medians_seconds':times,'contrasts':compare})
    means={arm:{m:sum(r['metrics'][arm][m] for r in pairs)/len(pairs) for m in ['f1','em']} for arm in cfg['arms']}
    summary={}
    fields={'context_changes':'context_changed','membership_changes':'membership_changed','output_token_changes':'output_tokens_changed','normalized_answer_changes':'normalized_answer_changed','full_ranking_changes':'full_ranking_changed','top10_ranking_changes':'top10_ranking_changed'}
    for name,a,b in contrasts:
        values=drifts[name]
        cost=latency_saving([r['latency_medians_seconds'][a] for r in pairs],[r['latency_medians_seconds'][b] for r in pairs],cfg['seed'],cfg['bootstrap_replicates'])
        cost['reference_sum_question_medians_seconds']=cost.pop('full_sum_question_medians_seconds');cost['treatment_sum_question_medians_seconds']=cost.pop('shortlist_sum_question_medians_seconds')
        summary[name]={'reference':a,'treatment':b,'paired':{m:interval([r['metrics'][b][m]-r['metrics'][a][m] for r in pairs],cfg['seed'],cfg['bootstrap_replicates']) for m in ['f1','em']},'latency_saving':cost,**{n:sum(r['contrasts'][name][field] for r in pairs) for n,field in fields.items()},'score_drift':{'unique_documents':len(values),'mean':float(np.mean(values)),'max':max(values),'p95':float(np.quantile(values,.95)),'above_1e4':sum(v>1e-4 for v in values)}}
    diagnostics={}
    for arm in cfg['arms']:
        vals=[r for r in requests if r['arm']==arm]
        diagnostics[arm]={'requests':len(vals),'scored_documents':sum(len(v['scores']) for v in vals),'forward_invocations':sum(v['forward_invocations'] for v in vals),'component_totals_seconds':{k:sum(v['seconds'][k] for v in vals) for k in vals[0]['seconds']},'repeat_totals_seconds':[sum(v['seconds']['total'] for v in vals if v['repeat']==rep) for rep in range(2)],'peak_allocated_bytes':max(v['peak_allocated_bytes'] for v in vals),'peak_reserved_bytes':max(v['peak_reserved_bytes'] for v in vals),'dtype_transition_seconds_excluded':sum(v['dtype_transition_seconds_excluded'] for v in vals),'padded_tokens':sum(v['padded_tokens'] for v in vals),'nonpad_tokens':sum(v['nonpad_tokens'] for v in vals),'cap_hits_first_repeat':sum(v['cap_hit'] for v in vals if v['repeat']==0)}
    primary=stability(pairs,details['engineering_gate_passed'],cfg)
    for name in summary:
        summary[name]['context_change_wilson95']=wilson(summary[name]['context_changes'],len(rows))
    native=summary['bf16_batch4_minus_bf16_serial']['score_drift']['mean']
    fp=summary['fp32_batch4_minus_fp32_serial']['score_drift']['mean']
    primary['descriptive_mean_drift_ratio_fp32_math_to_bf16_native']=fp/native if native else None
    release=json.loads((ROOT/cfg['data']/'label_release.json').read_text())
    assert release['labels_sha256']==file_sha(ROOT/cfg['labels']) and release['runtime_sha256']==file_sha(base/'runtime.json')
    gate=decision(cfg,details['engineering_gate_passed'],summary['fp32_batch4_minus_fp32_serial'],summary['fp32_batch4_minus_bf16_serial'])
    return pairs,{'methods':means,'contrasts':summary,'decision':gate,'primary_stability':primary,'diagnostics':diagnostics,'scope':cfg['scope'],'questions':len(rows),'primary_generation_calls':len(requests),'warmup_generation_calls':runtime['warmup_requests'],'primary_scored_documents':runtime['primary_score_calls'],'warmup_scored_documents':runtime['warmup_score_calls'],'scored_document_saving':0.0,'bootstrap':{'replicates':cfg['bootstrap_replicates'],'seed':cfg['seed'],'unit':'question','repetitions_are_not_extra_questions':True,'multiplicity_adjusted':False},'manifest_sha256':file_sha(base/'manifest.json'),'labels_sha256':file_sha(ROOT/cfg['labels']),'alias_aware_metric_sha256':file_sha(ROOT/'src/musique_metrics.py')}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','full','replay']);args=p.parse_args()
    cfg,base,m=check22();rows,ranks,schedule=source_check(cfg,base)
    toks=tokenizers(cfg) if args.mode!='replay' else {}
    if args.mode=='preflight':preflight(cfg,base,rows,toks)
    else:
        target=base/('analysis_remote' if args.mode=='full' else 'analysis_local');assert not target.exists()
        requests,grouped,runtime,details=audit(cfg,base,rows,ranks,schedule,toks)
        pairs,summary=evaluate(cfg,base,rows,requests,grouped,runtime,details)
        target.mkdir();write_new(target/'question_pairs.jsonl',pairs);atomic_json(target/'analysis.json',summary)
        atomic_json(target/'audit.json',{**details,'analysis_sha256':file_sha(target/'analysis.json'),'implementation_sha256':file_sha(__file__),'result_sha256':{v.name:file_sha(v) for v in base.iterdir() if v.is_file() and not v.name.startswith('queue_')}})
        print(json.dumps({'state':'audit_complete','audit':details,'methods':summary['methods'],'contrasts':summary['contrasts'],'decision':summary['decision']}),flush=True)
