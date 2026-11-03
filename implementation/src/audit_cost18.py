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
from cost18_core import check18
from audit_cqc_budget12 import chat_text, interval, norm, packed, request_messages, tokens_for_rerank


def reference_selection(row, ranks, arm, cfg):
    if arm == 'full':
        return list(range(len(row['documents'])))
    assert arm == 'rrf64'
    table = []
    for i in range(len(row['documents'])):
        value = fractions.Fraction(0, 1)
        for name in ('e5', 'bge'):
            rank = ranks[name][i]
            if rank is not None:
                assert isinstance(rank, int) and 1 <= rank <= cfg['retriever_depth']
                value += fractions.Fraction(1, cfg['rrf_constant'] + rank)
        table.append((value, i))
    table.sort(key=lambda pair: (-pair[0], pair[1]))
    return [i for _, i in table[:cfg['shortlist_size']]]


def source_check(cfg, base):
    rows = read_jsonl(base / 'inputs.jsonl')
    sources = read_jsonl(source_path(cfg['candidate_source']))
    excluded = read_jsonl(ROOT / cfg['excluded_source'])
    excluded_ids = {r['id'] for r in excluded}
    assert len(sources) == len({r['id'] for r in sources}) == 128
    assert len(excluded_ids) == len(excluded) == 32
    assert excluded_ids <= {r['id'] for r in sources}
    selected = [r for r in sources if r['id'] not in excluded_ids]
    expected = []
    for r in selected:
        indices = r['pools']['dual_dense_qwen']
        assert indices == sorted(set(indices))
        expected.append({'id': r['id'], 'question': r['question'], 'documents': [{k: r['documents'][i][k] for k in ['corpus_id','corpus_row','title','text']} for i in indices], 'source_document_indices': indices})
    assert rows == expected
    assert not excluded_ids.intersection(r['id'] for r in rows)
    assert len(rows) == len({r['id'] for r in rows}) == cfg['questions'] == 96
    assert sum(len(r['documents']) for r in rows) == 16699
    ranks_list = read_jsonl(base / 'retrieval_ranks.jsonl')
    assert [r['id'] for r in ranks_list] == [r['id'] for r in rows]
    sources = {n: {r['id']: r for r in read_jsonl(source_path(cfg[n + '_source']))} for n in ('bge', 'e5')}
    for row, ranking in zip(rows, ranks_list):
        assert set(row) == {'id', 'question', 'documents', 'source_document_indices'}
        assert all(set(d) == {'corpus_id', 'corpus_row', 'title', 'text'} for d in row['documents'])
        assert len({d['corpus_id'] for d in row['documents']}) == len(row['documents'])
        seen = set()
        for name in ('bge', 'e5'):
            original = sources[name][row['id']]
            assert original['question'] == row['question']
            retrieved = original['documents'][:cfg['retriever_depth']]
            ids = [d['corpus_id'] for d in retrieved]
            assert len(ids) == len(set(ids)) == cfg['retriever_depth']
            expected = [ids.index(d['corpus_id'])+1 if d['corpus_id'] in ids else None for d in row['documents']]
            assert ranking[name] == expected
            for d, rank in zip(row['documents'], expected):
                if rank is not None:
                    assert {k: retrieved[rank-1][k] for k in d} == d
            seen.update(ids)
        assert seen == {d['corpus_id'] for d in row['documents']}
    expected_schedule = []
    for repeat in range(cfg['repetitions']):
        ordered = sorted(rows, key=lambda r: digest({'seed': cfg['seed'], 'repeat': repeat, 'id': r['id']}))
        for pos, row in enumerate(ordered):
            original_index = next(i for i, r in enumerate(rows) if r['id'] == row['id'])
            arms = ['full', 'rrf64'] if (original_index + repeat) % 2 == 0 else ['rrf64', 'full']
            for slot, arm in enumerate(arms):
                expected_schedule.append({'repeat': repeat, 'position': pos, 'slot': slot, 'id': row['id'], 'arm': arm})
    schedule = read_jsonl(base / 'schedule.jsonl')
    assert schedule == expected_schedule
    for row in rows:
        first = [s['arm'] for s in schedule if s['id'] == row['id'] and s['slot'] == 0]
        assert first.count('full') == first.count('rrf64') == 1
    return rows, {r['id']: r for r in ranks_list}, schedule


def tokenizers(cfg):
    from transformers import AutoTokenizer
    result = {}
    for name, asset in [('reader', 'logs/qwen3_reader_stage10_assets_verified.json'), ('reranker', 'configs/cqc_budget_stage12_reranker_assets.json')]:
        record = json.loads((ROOT / asset).read_text())
        assert record['revision'] == cfg[name]['revision']
        folder = ROOT / 'models' / cfg[name]['repo'].replace('/', '--')
        for entry in record['files']:
            if not entry['name'].endswith('.safetensors'):
                assert file_sha(folder / entry['name']) == entry['sha256']
        result[name] = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
    return result


def preflight(cfg, base, rows, ranks, toks):
    maximum, truncated, prompt_bound = 0, 0, 0
    for row in rows:
        indices = reference_selection(row, ranks[row['id']], 'rrf64', cfg)
        assert len(indices) == len(set(indices)) == cfg['shortlist_size']
        for d in row['documents']:
            ids, cut = tokens_for_rerank(toks['reranker'], row['question'], d, cfg['reranker_max_length'])
            maximum = max(maximum, len(ids))
            truncated += cut
        empty = chat_text(request_messages(row['question'], '', {}, True))
        bound = len(toks['reader'].encode(empty, add_special_tokens=False)) + cfg['context_tokens'] + 8
        assert bound <= cfg['generator_max_input_tokens']
        prompt_bound = max(prompt_bound, bound)
    result = {'state': 'passed', 'manifest_sha256': file_sha(base / 'manifest.json'), 'questions': len(rows), 'max_reranker_tokens': maximum, 'truncated_unique_original_inputs': truncated, 'conservative_prompt_bound': prompt_bound, 'labels_parsed': False, 'gpu_model_loaded': False}
    atomic_json(base / 'preflight.json', result)
    print(json.dumps(result), flush=True)


def latency_saving(full, shortlist, seed, replicates):
    a, b = np.asarray(full, dtype=float), np.asarray(shortlist, dtype=float)
    assert a.shape == b.shape and a.ndim == 1 and len(a) and np.isfinite(a).all() and np.isfinite(b).all() and (a > 0).all() and (b > 0).all()
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, replicates, 1000):
        sample = rng.integers(len(a), size=(min(1000, replicates-start), len(a)))
        values.extend((1 - b[sample].sum(axis=1) / a[sample].sum(axis=1)).tolist())
    return {'n': len(a), 'point': float(1-b.sum()/a.sum()), 'ci95': np.quantile(values, [.025, .975]).tolist(), 'full_sum_question_medians_seconds': float(a.sum()), 'shortlist_sum_question_medians_seconds': float(b.sum())}


def gate(cfg, engineering, paired, cost, calls):
    quality = paired['f1']['ci95'][0] >= cfg['minimum_f1_ci_lower'] and paired['em']['ci95'][0] >= cfg['minimum_em_ci_lower']
    latency = cost['point'] >= cfg['minimum_latency_saving'] and cost['ci95'][0] > 0
    count = calls >= cfg['minimum_score_call_saving']
    return {'engineering_passed': bool(engineering), 'quality_screen_passed': bool(quality), 'latency_screen_passed': bool(latency), 'score_call_screen_passed': bool(count), 'pilot_screen_passed': bool(engineering and quality and latency and count), 'reserved_expansion_authorized': False}


def verify_prior_scores(cfg, base, rows):
    runtime = json.loads(source_path(cfg['score_runtime']).read_text())
    originals = {r['id']: r for r in read_jsonl(source_path(cfg['candidate_source']))}
    cached = {}
    for name, expected in runtime['cache_sources'].items():
        assert file_sha(source_path(name)) == expected
        for item in read_jsonl(source_path(name)):
            if item['key'] in cached:
                assert item['score'] == cached[item['key']]['score']
            cached[item['key']] = item
    saved = read_jsonl(base / 'audit_only_prior_scores.jsonl')
    assert [r['id'] for r in saved] == [r['id'] for r in rows]
    result = {}
    for row, reference in zip(rows, saved):
        values = []
        for document in row['documents']:
            request = {'question': row['question'], 'passages': [{'title': document['title'], 'text': document['text']}]}
            key = digest({'input': request, 'scorer': runtime['scorer_hash']})
            item = cached[key]
            assert item['question_id'] == row['id'] and item['corpus_id'] == document['corpus_id']
            values.append(item['score'])
        assert values == reference['scores']
        order = sorted(range(len(values)), key=lambda i: (-values[i], i))
        assert [row['source_document_indices'][i] for i in order] == originals[row['id']]['rankings']['dual_dense_qwen']
        result[row['id']] = values
    return result


def pooled_quality(base, pairs, cfg):
    old = read_jsonl(ROOT / 'results/cost_stage17_pilot32/analysis_local/question_pairs.jsonl')
    assert len(old) == 32 and len(pairs) == 96
    all_pairs = old + pairs
    assert len({r['id'] for r in all_pairs}) == 128
    return {'scope': 'Descriptive pooled used128, includes the Stage17 design/pilot panel. Not a replacement for the primary96 gate; no pooled inferential advance decision.', 'n': 128, 'methods': {arm: {metric: sum(r['metrics'][arm][metric] for r in all_pairs) / len(all_pairs) for metric in ['f1','em']} for arm in cfg['arms']}}


def audit(cfg, base, rows, ranks, schedule, toks):
    runtime = json.loads((base / 'runtime.json').read_text())
    assert runtime['state'] == 'complete' and runtime['manifest_sha256'] == file_sha(base / 'manifest.json')
    assert runtime['implementation_sha256'] == file_sha(ROOT / 'src/run_cost18.py')
    assert not runtime['answer_result_cache'] and not runtime['reranker_result_cache']
    for name, sha in runtime['output_sha256'].items():
        assert file_sha(base / name) == sha
    requests = read_jsonl(base / 'requests.jsonl')
    warmup = read_jsonl(base / 'warmup.jsonl')
    assert [{k: r[k] for k in ['repeat','position','slot','id','arm']} for r in requests] == schedule
    assert [(r['id'],r['arm']) for r in warmup] == [(r['id'], arm) for r in rows[:cfg['warmup_questions']] for arm in cfg['arms']]
    assert len(requests) == runtime['primary_requests'] == 384
    assert len(warmup) == runtime['warmup_requests'] == 4
    assert sum(len(r['scores']) for r in requests) == runtime['primary_score_calls'] == 45686
    assert sum(len(r['scores']) for r in warmup) == runtime['warmup_score_calls']
    by_id = {r['id']: r for r in rows}
    old_scores = verify_prior_scores(cfg, base, rows)
    encoded = {}
    max_prior_difference = 0.
    for r in warmup + requests:
        row = by_id[r['id']]
        indices = reference_selection(row, ranks[r['id']], r['arm'], cfg)
        assert r['scored_indices'] == indices
        assert len(indices) == len(r['scores']) == len(r['reranker_inputs'])
        assert all(math.isfinite(v) for v in r['scores'])
        mapped = dict(zip(indices, r['scores']))
        assert r['order'] == sorted(indices, key=lambda i: (-mapped[i], i))
        for index, value, evidence in zip(indices, r['scores'], r['reranker_inputs']):
            max_prior_difference = max(max_prior_difference, abs(value-old_scores[r['id']][index]))
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
    repeats = []
    max_repeat_difference = 0.
    for (qid, arm), values in grouped.items():
        assert [r['repeat'] for r in values] == list(range(cfg['repetitions']))
        for other in values[1:]:
            difference = max(abs(a-b) for a,b in zip(values[0]['scores'],other['scores']))
            max_repeat_difference = max(max_repeat_difference, difference)
            if other['output_ids'] != values[0]['output_ids'] or other['packing'] != values[0]['packing'] or other['reranker_inputs'] != values[0]['reranker_inputs']:
                repeats.append({'id': qid, 'arm': arm, 'repeat': other['repeat']})
    engineering = not repeats and max_repeat_difference <= cfg['reranker_repeat_atol'] and max_prior_difference <= cfg['reranker_repeat_atol']
    details = {'state': 'passed', 'tokenization_reconstructed': bool(toks), 'engineering_gate_passed': engineering, 'primary_requests': len(requests), 'warmup_requests': len(warmup), 'primary_score_calls': runtime['primary_score_calls'], 'warmup_score_calls': runtime['warmup_score_calls'], 'max_stage09_original_score_difference': max_prior_difference, 'max_repeat_score_difference': max_repeat_difference, 'repeat_mismatches': repeats, 'prior_reader_answer_replay_checked': False, 'labels_parsed_during_structural_audit': False, 'manifest_sha256': file_sha(base / 'manifest.json')}
    return requests, grouped, runtime, details


def evaluate(cfg, base, rows, requests, grouped, runtime, details):
    loader = importlib.machinery.SourceFileLoader('hotpot_cost18', str(ROOT / 'references/hotpot_eval.txt'))
    evaluator = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(evaluator)
    labels = {r['id']: r for r in read_jsonl(ROOT / 'data/stage08_confirmation/labels.jsonl')}
    pairs = []
    for row in rows:
        label = labels[row['id']]
        assert label['question'] == row['question']
        metrics, medians = {}, {}
        first = {arm: grouped[(row['id'],arm)][0] for arm in cfg['arms']}
        for arm in cfg['arms']:
            answer = first[arm]['answer']
            metrics[arm] = {'answer': answer, 'f1': float(evaluator.f1_score(answer, label['answer'])[0]), 'em': float(evaluator.exact_match_score(answer,label['answer']))}
            medians[arm] = float(np.median([r['seconds']['total'] for r in grouped[(row['id'],arm)]]))
        pairs.append({'id': row['id'], 'metrics': metrics, 'latency_medians_seconds': medians, 'full_candidates': len(row['documents']), 'context_changed': first['full']['packing']['context'] != first['rrf64']['packing']['context'], 'normalized_answer_changed': norm(first['full']['answer']) != norm(first['rrf64']['answer']), 'full_selected_retained_in_shortlist': len(set(first['full']['packing']['selected']) & set(first['rrf64']['scored_indices'])), 'full_selected_count': len(first['full']['packing']['selected'])})
    means = {arm: {m: sum(r['metrics'][arm][m] for r in pairs)/len(pairs) for m in ['f1','em']} for arm in cfg['arms']}
    paired = {m: interval([r['metrics']['rrf64'][m]-r['metrics']['full'][m] for r in pairs], cfg['seed'], cfg['bootstrap_replicates']) for m in ['f1','em']}
    saving = latency_saving([r['latency_medians_seconds']['full'] for r in pairs], [r['latency_medians_seconds']['rrf64'] for r in pairs], cfg['seed'], cfg['bootstrap_replicates'])
    counts = {arm: sum(len(r['scores']) for r in requests if r['arm']==arm) for arm in cfg['arms']}
    call_saving = 1-counts['rrf64']/counts['full']
    diagnostics = {}
    for arm in cfg['arms']:
        arm_rows = [r for r in requests if r['arm']==arm]
        times = [r['seconds']['total'] for r in arm_rows]
        diagnostics[arm] = {'raw_requests': len(times), 'all_requests_median_seconds': float(np.median(times)), 'all_requests_p95_seconds': float(np.quantile(times,.95)), 'component_totals_seconds': {k: sum(r['seconds'][k] for r in arm_rows) for k in arm_rows[0]['seconds']}, 'repeat_totals_seconds': [sum(r['seconds']['total'] for r in arm_rows if r['repeat']==rep) for rep in range(cfg['repetitions'])], 'slot_medians_seconds': {str(slot): float(np.median([r['seconds']['total'] for r in arm_rows if r['slot']==slot])) for slot in [0,1]}, 'cap_hits_first_repeat': sum(r['cap_hit'] for r in arm_rows if r['repeat']==0)}
    decision = gate(cfg, details['engineering_gate_passed'], paired, saving, call_saving)
    summary = {'methods': means, 'primary_contrast': 'rrf64_minus_full', 'paired': paired, 'latency_saving': saving, 'primary_score_calls': counts, 'score_call_saving': call_saving, 'primary_generation_calls': len(requests), 'warmup_generation_calls': runtime['warmup_requests'], 'warmup_score_calls': runtime['warmup_score_calls'], 'context_changes': sum(r['context_changed'] for r in pairs), 'normalized_answer_changes': sum(r['normalized_answer_changed'] for r in pairs), 'decision': decision, 'latency_diagnostics': diagnostics, 'scope': cfg['scope'], 'bootstrap': {'replicates': cfg['bootstrap_replicates'], 'seed': cfg['seed'], 'unit': 'question', 'repeated_timings_are_not_extra_questions': True, 'timing_repetitions_per_question_arm': cfg['repetitions'], 'scope': 'Question bootstrap conditional on observed two-repeat medians; not complete timing-noise uncertainty', 'multiplicity_adjusted': False}, 'manifest_sha256': file_sha(base/'manifest.json'), 'labels_sha256': file_sha(ROOT/'data/stage08_confirmation/labels.jsonl'), 'official_evaluator_sha256': file_sha(ROOT/'references/hotpot_eval.txt')}
    return pairs, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['preflight','full','replay'])
    args = parser.parse_args()
    cfg, base, _ = check18()
    rows, ranks, schedule = source_check(cfg, base)
    toks = tokenizers(cfg) if args.mode != 'replay' else {}
    if args.mode == 'preflight':
        preflight(cfg,base,rows,ranks,toks)
    else:
        destination = base/('analysis_remote' if args.mode=='full' else 'analysis_local')
        assert not destination.exists()
        requests, grouped, runtime, details = audit(cfg,base,rows,ranks,schedule,toks)
        pairs, summary = evaluate(cfg,base,rows,requests,grouped,runtime,details)
        summary['pooled_used128_descriptive'] = pooled_quality(base, pairs, cfg)
        destination.mkdir()
        write_new(destination/'question_pairs.jsonl', pairs)
        atomic_json(destination/'analysis.json', summary)
        atomic_json(destination/'audit.json', {**details, 'analysis_sha256': file_sha(destination/'analysis.json'), 'implementation_sha256': file_sha(__file__), 'result_sha256': {p.name: file_sha(p) for p in base.iterdir() if p.is_file()}})
        print(json.dumps({'state': 'passed', 'details': details, 'methods': summary['methods'], 'paired': summary['paired'], 'latency_saving': summary['latency_saving'], 'decision': summary['decision']}), flush=True)
