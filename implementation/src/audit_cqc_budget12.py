import argparse
import collections
import hashlib
import importlib.machinery
import importlib.util
import itertools
import json
import math
import re
import string

import numpy as np

from common import ROOT, atomic_json, digest, read_jsonl
from cqc_budget12_core import file_sha, settings, source_path, write_new


RULES = ['cqc_paper_var', 'cqc_code_std', 'cross_mean', 'own_verifier', 'original_query_verifier', 'vote']
CONTROLS = ['view0', 'original_neutral', 'maxscore_neutral', 'maxscore_evidence']
CONDITIONS = ['view' + str(i) for i in range(5)] + CONTROLS[1:]
PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def norm(text):
    value = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', value).split())


def paraphrases(raw, original):
    items = []
    for line in raw.splitlines():
        match = re.match(r'^\s*[0-9]+[.)]\s*(.+)', line)
        if match:
            value = match[1].strip()
            items.append(value if value.endswith('?') else value + '?')
    return [original] + items[:4] + [original] * max(0, 4 - len(items)), len(items), max(0, 4 - len(items))


def parse(raw, neutral=False):
    if neutral:
        return {'answer': raw, 'evidence': '', 'format_has_evidence': False}
    parts = re.split(r'evidence:\s*', raw.strip(), maxsplit=1, flags=re.I)
    return {'answer': re.sub(r'^answer:\s*', '', parts[0].strip(), flags=re.I).strip(), 'evidence': parts[1].strip() if len(parts) == 2 else '', 'format_has_evidence': len(parts) == 2}


def independent_rules(candidates, matrix, coefficient=.5):
    p = np.asarray(matrix, dtype=np.float64)
    assert p.shape == (5, 5) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    means = p.mean(axis=1)
    var = p.var(axis=1, ddof=0)
    values = {'cqc_paper_var': means - coefficient * means * var, 'cqc_code_std': np.clip(means - np.where(means > .7, .5, .2) * np.sqrt(var), 0, 1), 'cross_mean': means, 'own_verifier': p.diagonal(), 'original_query_verifier': p[:, 0]}
    counts = collections.Counter(norm(r['answer']) for r in candidates)
    values['vote'] = np.asarray([counts[norm(r['answer'])] for r in candidates], dtype=float)
    eligible = np.asarray([bool(r['answer']) for r in candidates])
    scores = {name: np.where(eligible, values[name], -1e9).tolist() for name in RULES}
    return {name: int(np.argmax(scores[name])) for name in RULES}, scores


def interval(values, seed=20261013, replicates=20000):
    array = np.asarray(values, dtype=float)
    assert array.ndim == 1 and len(array) and np.isfinite(array).all()
    rng = np.random.default_rng(seed)
    samples = np.concatenate([array[rng.integers(len(array), size=(min(1000, replicates - k), len(array)))].mean(axis=1) for k in range(0, replicates, 1000)])
    return {'n': len(array), 'mean': float(array.mean()), 'ci95': np.quantile(samples, [.025, .975]).tolist(), 'positive': int((array > 1e-12).sum()), 'negative': int((array < -1e-12).sum()), 'equal': int((np.abs(array) <= 1e-12).sum())}


def indexed(path, keys):
    rows = read_jsonl(path)
    result = {tuple(r[k] for k in keys): r for r in rows}
    assert len(result) == len(rows), ('duplicate rows', str(path))
    return result


def frozen_inputs(config, base):
    manifest = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files_sha256'].items():
        assert file_sha(ROOT / name) == expected, name
    assert manifest['generation_started_at_freeze'] is False and manifest['labels_in_inputs'] is False
    assert manifest['source_path'] == config['candidate_source']
    source = source_path(manifest['source_path'])
    assert file_sha(source) == manifest['source_sha256']
    assert file_sha(base / 'inputs.jsonl') == manifest['inputs_sha256']
    source_rows = read_jsonl(source)
    used_rows = [r for r in read_jsonl(ROOT / 'results/reader_replay_stage10/inputs.jsonl') if r['dataset'] == 'hotpot']
    used = {r['id']: r['question'] for r in used_rows}
    assert len(used) == len(source_rows) == 128
    assert all(used[r['id']] == r['question'] for r in used_rows)
    assert {r['id'] for r in source_rows} == set(used)
    ordered = sorted(source_rows, key=lambda r: hashlib.sha256((config['sample_key'] + r['id']).encode()).hexdigest())[:config['questions']]
    rows = read_jsonl(base / 'inputs.jsonl')
    assert [r['id'] for r in rows] == manifest['ids'] == [r['id'] for r in ordered]
    assert len(rows) == manifest['questions'] == config['questions'] == 32
    for row, original in zip(rows, ordered):
        assert set(row) == {'id', 'question', 'documents', 'source_document_indices'}
        assert row['question'] == original['question'] == used[row['id']]
        indices = original['pools'][config['pool']]
        assert indices == row['source_document_indices'] and len(indices) == len(set(indices))
        projected = [{k: original['documents'][i][k] for k in ['corpus_id', 'corpus_row', 'title', 'text']} for i in indices]
        assert row['documents'] == projected
        assert len({d['corpus_id'] for d in projected}) == len(projected)
    assert manifest['pool_counts'] == [len(r['documents']) for r in rows]
    return rows, manifest


def tokens_for_rerank(tokenizer, query, document, limit=1024):
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    text = '<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: ' + query + '\n<Document>: Title: ' + document['title'] + '\n' + document['text']
    body = tokenizer.encode(text, add_special_tokens=False)
    available = limit - len(prefix) - len(suffix)
    assert available > 0
    return prefix + body[:available] + suffix, len(body) > available


def packed(documents, order, tokenizer, budget, topk):
    chosen = []
    for index in order[:topk]:
        trial = chosen + [index]
        text = '\n\n'.join('Title: ' + documents[i]['title'] + '\n' + documents[i]['text'] for i in trial)
        if len(tokenizer.encode(text, add_special_tokens=False)) <= budget:
            chosen.append(index)
    text = '\n\n'.join('Title: ' + documents[i]['title'] + '\n' + documents[i]['text'] for i in chosen)
    return {'selected': chosen, 'context': text, 'context_tokens': len(tokenizer.encode(text, add_special_tokens=False))}


def yes_no(tokenizer):
    result = {}
    for word in ['Yes', 'No']:
        plain = tokenizer.encode(word, add_special_tokens=False)
        spaced = tokenizer.encode(' ' + word, add_special_tokens=False)
        assert min(len(plain), len(spaced)) == 1
        result[word] = {'plain': plain, 'spaced': spaced, 'chosen': spaced[0] if len(spaced) == 1 else plain[0]}
    assert result['Yes']['chosen'] != result['No']['chosen']
    return result


def load_tokenizers(config):
    from transformers import AutoTokenizer
    result = {}
    for name in ['reader', 'reranker', 'evaluator']:
        detail = config[name]
        if name == 'reader':
            record = json.loads((ROOT / 'logs/qwen3_reader_stage10_assets_verified.json').read_text())
        elif name == 'evaluator':
            record = json.loads((ROOT / 'logs/second_reader_assets_verified.json').read_text())
        else:
            record = json.loads((ROOT / 'configs/cqc_budget_stage12_reranker_assets.json').read_text())
        assert record['revision'] == detail['revision']
        folder = ROOT / 'models' / detail['repo'].replace('/', '--')
        for entry in record['files']:
            if not entry['name'].endswith('.safetensors'):
                assert file_sha(folder / entry['name']) == entry['sha256'], entry['name']
        result[name] = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
    return result


def chat_text(messages):
    return ''.join('<|im_start|>' + r['role'] + '\n' + r['content'] + '<|im_end|>\n' for r in messages) + '<|im_start|>assistant\n<think>\n\n</think>\n\n'


def request_messages(query, context, prompts, neutral=False):
    if neutral:
        return [{'role': 'system', 'content': 'You are a helpful question answering assistant.'}, {'role': 'user', 'content': 'Answer the question using the provided passages. Output only the short answer, with no explanation.\n\nPassages:\n' + context + '\n\nQuestion: ' + query}]
    return [{'role': 'system', 'content': prompts['answer_system']}, {'role': 'user', 'content': 'Documents:\n' + context + '\n\nQuestion: ' + query}]


def generated(cache, key, messages, cap, sampled, seed, config, freeze, tokenizer, eos):
    options = {'do_sample': sampled, 'max_new_tokens': cap, 'pad_token_id': eos, 'repetition_penalty': 1.0}
    if sampled:
        options.update(temperature=config['paraphrase_temperature'], top_p=config['paraphrase_top_p'], top_k=0)
    text = chat_text(messages)
    expected_key = digest({'prompt': text, 'options': options, 'seed': seed if sampled else None, 'freeze': freeze})
    assert key == expected_key
    row = cache[(key,)]
    assert row['sampling_implementation'] == ('temperature_then_cpu_top_p_seeded_cuda_multinomial' if sampled else 'greedy')
    assert row['prompt'] == text and row['options'] == options and row['seed'] == (seed if sampled else None)
    assert 0 < len(row['output_ids']) <= cap and row['cap_hit'] == (len(row['output_ids']) == cap)
    assert row['input_tokens'] <= config['generator_max_input_tokens'] and row['seconds'] >= 0
    if tokenizer is not None:
        assert text == tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer(text, add_special_tokens=False).input_ids
        assert ids == tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        assert digest(ids) == row['input_ids_sha256'] and len(ids) == row['input_tokens']
        assert tokenizer.decode(row['output_ids'], skip_special_tokens=True).strip() == row['raw']
    return row


def preflight(config, base, inputs, tokenizers, prompts):
    reader, reranker, evaluator = [tokenizers[k] for k in ['reader', 'reranker', 'evaluator']]
    maxima = {'paraphrase_prompt': 0, 'answer_prompt_with_original_query': 0, 'reranker_input': 0}
    truncated = 0
    for row in inputs:
        messages = [{'role': 'system', 'content': prompts['paraphrase_system']}, {'role': 'user', 'content': 'Original: ' + row['question']}]
        ids = reader.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        assert reader.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False) == chat_text(messages)
        assert len(ids) <= config['generator_max_input_tokens']
        maxima['paraphrase_prompt'] = max(maxima['paraphrase_prompt'], len(ids))
        context = packed(row['documents'], list(range(len(row['documents']))), reader, config['context_tokens'], config['top_k'])['context']
        for neutral in [False, True]:
            messages = request_messages(row['question'], context, prompts, neutral)
            ids = reader.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            assert len(ids) <= config['generator_max_input_tokens']
            maxima['answer_prompt_with_original_query'] = max(maxima['answer_prompt_with_original_query'], len(ids))
        for document in row['documents']:
            ids, cut = tokens_for_rerank(reranker, row['question'], document)
            maxima['reranker_input'] = max(maxima['reranker_input'], len(ids))
            truncated += cut
        prompt = prompts['verification_template'].format(question=row['question'], answer='', evidence='')
        ids = evaluator(prompt, add_special_tokens=True).input_ids
        assert len(ids) <= config['verifier_max_input_tokens'] and ids[0] == evaluator.bos_token_id and ids[1] != evaluator.bos_token_id
    value = {'state': 'passed', 'questions': len(inputs), 'documents': sum(len(r['documents']) for r in inputs), 'max_tokens': maxima, 'original_query_truncated_reranker_inputs': truncated, 'binary_token_ids': yes_no(evaluator), 'labels_read': False, 'gpu_model_loaded': False, 'manifest_sha256': file_sha(base / 'manifest.json'), 'scope': 'All original-query pool encodings and static prompt checks. Generated paraphrase/answer/evidence lengths are checked at their later stages, not predicted here.'}
    atomic_json(base / 'preflight.json', value)
    print(json.dumps(value), flush=True)


def check_outputs(config, base, inputs, tokenizers, prompt_config):
    freeze = file_sha(base / 'manifest.json')
    rt = {stage: json.loads((base / (stage + '_runtime.json')).read_text()) for stage in ['paraphrase', 'rerank', 'answers', 'verify']}
    caches = {stage: indexed(base / name, ['key']) for stage, name in [('paraphrase', 'paraphrase_cache.jsonl'), ('rerank', 'reranker_cache.jsonl'), ('answers', 'answer_cache.jsonl'), ('verify', 'verifier_cache.jsonl')]}
    names = {'paraphrase': 'paraphrase_cache.jsonl', 'rerank': 'reranker_cache.jsonl', 'answers': 'answer_cache.jsonl', 'verify': 'verifier_cache.jsonl'}
    for stage, runtime in rt.items():
        assert runtime['state'] == 'complete' and runtime['manifest_sha256'] == freeze
        assert runtime['implementation_sha256'] == file_sha(ROOT / 'src/run_cqc_budget12.py')
        assert runtime['cache_sha256'] == file_sha(base / names[stage])
        assert runtime['cache_entries'] == len(caches[stage])
        assert 0 <= runtime['new_calls'] <= runtime['cache_entries']
        assert runtime['rows'] == len(inputs) * {'paraphrase': 1, 'rerank': 1, 'answers': 8, 'verify': 25}[stage]
    reader = tokenizers.get('reader')
    scorer_config = json.loads((ROOT / 'configs/qwen512_v1.json').read_text())
    assert scorer_config['max_length'] == 1024 and scorer_config['base_model'] == config['reranker']['repo'] and scorer_config['base_revision'] == config['reranker']['revision']
    queries = indexed(base / 'queries.jsonl', ['id'])
    rankings = indexed(base / 'rankings.jsonl', ['id'])
    answers = indexed(base / 'answers.jsonl', ['id', 'condition'])
    verification = indexed(base / 'verification.jsonl', ['id', 'candidate_view', 'query_view'])
    ids = {r['id'] for r in inputs}
    assert set(queries) == set(rankings) == {(i,) for i in ids}
    assert set(answers) == {(i, c) for i in ids for c in CONDITIONS}
    assert set(verification) == {(i, j, k) for i in ids for j in range(5) for k in range(5)}
    accessed = {stage: set() for stage in rt}
    decisions = []
    diagnostics = []
    token_ids = rt['verify']['binary_token_ids']
    if tokenizers:
        assert token_ids == yes_no(tokenizers['evaluator'])
    yes, no = token_ids['Yes']['chosen'], token_ids['No']['chosen']
    eos = rt['paraphrase']['reader_eos_token_id']
    assert eos == rt['answers']['reader_eos_token_id']
    if reader:
        assert eos == reader.eos_token_id
    from cqc_budget12_core import select_rules
    truncations = 0
    for row in inputs:
        qid = row['id']
        q = queries[(qid,)]
        seed = int(hashlib.sha256((str(config['seed']) + '|' + qid).encode()).hexdigest()[:8], 16)
        messages = [{'role': 'system', 'content': prompt_config['paraphrase_system']}, {'role': 'user', 'content': 'Original: ' + row['question']}]
        cache = generated(caches['paraphrase'], q['key'], messages, config['paraphrase_max_new_tokens'], True, seed, config, freeze, reader, eos)
        accessed['paraphrase'].add((q['key'],))
        reconstructed, count, fallback = paraphrases(cache['raw'], row['question'])
        assert q['original'] == row['question'] and q['queries'] == reconstructed
        assert q['raw'] == cache['raw'] and q['cap_hit'] == cache['cap_hit'] and q['parsed_count'] == count and q['fallback_count'] == fallback
        rank = rankings[(qid,)]
        assert rank['pool_corpus_ids'] == [d['corpus_id'] for d in row['documents']]
        assert [v['view'] for v in rank['views']] == list(range(5))
        for j, view in enumerate(rank['views']):
            assert view['query'] == q['queries'][j]
            assert len(view['scores']) == len(view['score_keys']) == len(row['documents'])
            for k, document in enumerate(row['documents']):
                blind = {'question': view['query'], 'passages': [{'title': document['title'], 'text': document['text']}]}
                key = digest({'input': blind, 'freeze': freeze, 'scorer_config': digest(scorer_config)})
                assert key == view['score_keys'][k]
                saved = caches['rerank'][(key,)]
                accessed['rerank'].add((key,))
                assert saved['query'] == view['query'] and saved['document'] == blind['passages'][0]
                assert saved['score'] == view['scores'][k] and math.isfinite(saved['score'])
                assert 0 < saved['input_tokens'] <= 1024
                truncations += saved['truncated']
                if tokenizers:
                    encoded, cut = tokens_for_rerank(tokenizers['reranker'], view['query'], document, scorer_config['max_length'])
                    assert len(encoded) == saved['input_tokens'] and digest(encoded) == saved['input_ids_sha256'] and cut == saved['truncated']
            assert view['order'] == sorted(range(len(row['documents'])), key=lambda k: (-view['scores'][k], k))
        maximum = [max(v['scores'][k] for v in rank['views']) for k in range(len(row['documents']))]
        assert maximum == rank['maxscore']['scores']
        assert rank['maxscore']['order'] == sorted(range(len(maximum)), key=lambda k: (-maximum[k], k))
        for view in rank['views'] + [rank['maxscore']]:
            assert len(set(view['selected'])) == len(view['selected'])
            assert view['selected'] == [i for i in view['order'][:config['top_k']] if i in view['selected']]
            context = '\n\n'.join('Title: ' + row['documents'][i]['title'] + '\n' + row['documents'][i]['text'] for i in view['selected'])
            assert context == view['context'] and 0 <= view['context_tokens'] <= config['context_tokens']
            if reader:
                expected = packed(row['documents'], view['order'], reader, config['context_tokens'], config['top_k'])
                assert all(view[k] == v for k, v in expected.items())
        for condition in CONDITIONS:
            view_index = int(condition[4:]) if condition.startswith('view') else None
            query = q['queries'][view_index] if view_index is not None else row['question']
            context = rank['views'][view_index]['context'] if view_index is not None else rank['views'][0]['context'] if condition == 'original_neutral' else rank['maxscore']['context']
            neutral = condition.endswith('_neutral')
            answer = answers[(qid, condition)]
            assert answer['query'] == query and answer['context_sha256'] == digest(context)
            cap = config['neutral_max_new_tokens'] if neutral else config['answer_max_new_tokens']
            messages = request_messages(query, context, prompt_config, neutral)
            saved = generated(caches['answers'], answer['key'], messages, cap, False, config['seed'], config, freeze, reader, eos)
            accessed['answers'].add((answer['key'],))
            assert answer['raw'] == saved['raw'] and answer['cap_hit'] == saved['cap_hit']
            assert all(answer[k] == v for k, v in parse(saved['raw'], neutral).items())
            evidence_words = set(norm(answer['evidence']).split())
            context_words = set(norm(context).split())
            diagnostics.append({'id': qid, 'condition': condition, 'empty_answer': not bool(answer['answer']), 'empty_evidence': not bool(answer['evidence']), 'format_has_evidence': answer['format_has_evidence'], 'empty_context': not bool(context), 'cap_hit': answer['cap_hit'], 'evidence_normalized_substring': bool(norm(answer['evidence'])) and norm(answer['evidence']) in norm(context), 'evidence_unique_word_coverage': len(evidence_words & context_words) / len(evidence_words) if evidence_words else None})
        matrix = []
        candidates = [answers[(qid, 'view' + str(i))] for i in range(5)]
        for i, candidate in enumerate(candidates):
            probabilities = []
            for j, query in enumerate(q['queries']):
                prompt = prompt_config['verification_template'].format(question=query, answer=candidate['answer'], evidence=candidate['evidence'])
                key = digest({'prompt': prompt, 'yes': yes, 'no': no, 'freeze': freeze})
                saved = caches['verify'][(key,)]
                accessed['verify'].add((key,))
                assert verification[(qid, i, j)]['key'] == key and saved['prompt'] == prompt
                assert math.isfinite(saved['yes_logit']) and math.isfinite(saved['no_logit'])
                difference = saved['yes_logit'] - saved['no_logit']
                probability = 1 / (1 + math.exp(-difference)) if difference >= 0 else math.exp(difference) / (1 + math.exp(difference))
                assert abs(saved['probability'] - probability) < 1e-7
                assert verification[(qid, i, j)]['probability'] == saved['probability']
                assert 0 < saved['input_tokens'] <= config['verifier_max_input_tokens']
                if tokenizers:
                    tokenizer = tokenizers['evaluator']
                    encoded = tokenizer(prompt, add_special_tokens=True).input_ids
                    assert encoded[0] == tokenizer.bos_token_id and encoded[1] != tokenizer.bos_token_id
                    assert digest(encoded) == saved['input_ids_sha256'] and len(encoded) == saved['input_tokens']
                probabilities.append(saved['probability'])
            matrix.append(probabilities)
        choices, scores = independent_rules(candidates, matrix, config['paper_lambda0'])
        production_choices, production_scores = select_rules(candidates, matrix, config['paper_lambda0'])
        assert choices == production_choices
        assert all(np.allclose(scores[k], production_scores[k], rtol=0, atol=1e-12) for k in RULES)
        decisions.append({'id': qid, 'choices': choices, 'scores': scores, 'matrix': matrix, 'answers': {c: answers[(qid, c)]['answer'] for c in CONDITIONS}, 'paraphrase_diagnostics': {'parsed_count': count, 'fallback_count': fallback, 'cap_hit': q['cap_hit'], 'duplicate_extra_views_exact': 5 - len(set(q['queries'])), 'duplicate_extra_views_normalized': 5 - len(set(norm(v) for v in q['queries']))}})
    assert truncations == rt['rerank']['truncated_condition_inputs']
    assert rt['rerank']['condition_pairs'] == sum(len(r['documents']) * 5 for r in inputs)
    for stage in rt:
        assert accessed[stage] == set(caches[stage]), ('unused or missing cache keys', stage)
    return decisions, diagnostics, rt, {stage: len(cache) for stage, cache in caches.items()}


def evaluate(config, base, inputs, decisions, diagnostics, runtimes, cache_counts):
    loader = importlib.machinery.SourceFileLoader('hotpot_cqc12', str(ROOT / 'references/hotpot_eval.txt'))
    evaluator = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(evaluator)
    labels = {r['id']: r for r in read_jsonl(ROOT / 'data/stage08_confirmation/labels.jsonl')}
    methods = RULES + CONTROLS
    scored = []
    for input_row, selection in zip(inputs, decisions):
        label = labels[input_row['id']]
        assert label['question'] == input_row['question']
        metrics = {}
        for method in methods:
            condition = 'view' + str(selection['choices'][method]) if method in RULES else method
            prediction = selection['answers'][condition]
            metrics[method] = {'prediction': prediction, 'condition': condition, 'em': float(evaluator.exact_match_score(prediction, label['answer'])), 'f1': float(evaluator.f1_score(prediction, label['answer'])[0])}
        candidate_metrics = {c: {'em': float(evaluator.exact_match_score(selection['answers'][c], label['answer'])), 'f1': float(evaluator.f1_score(selection['answers'][c], label['answer'])[0])} for c in CONDITIONS}
        assert all(norm(a) == evaluator.normalize_answer(a) for a in selection['answers'].values())
        scored.append({**selection, 'metrics': metrics, 'candidate_metrics': candidate_metrics})
    summary = {m: {metric: float(np.mean([r['metrics'][m][metric] for r in scored])) for metric in ['f1', 'em']} for m in methods}
    comparisons = {}
    for left, right in itertools.combinations(methods, 2):
        comparisons[left + '__minus__' + right] = {metric: interval([r['metrics'][left][metric] - r['metrics'][right][metric] for r in scored], config['seed']) for metric in ['f1', 'em']}
    disagreements = {left + '__vs__' + right: sum(r['choices'][left] != r['choices'][right] for r in scored) for left, right in itertools.combinations(RULES, 2)}
    formatting = {}
    for c in CONDITIONS:
        rows = [r for r in diagnostics if r['condition'] == c]
        formatting[c] = {k: sum(r[k] for r in rows) for k in ['empty_answer', 'empty_evidence', 'format_has_evidence', 'empty_context', 'cap_hit', 'evidence_normalized_substring']}
        coverage = [r['evidence_unique_word_coverage'] for r in rows if r['evidence_unique_word_coverage'] is not None]
        formatting[c]['evidence_mean_unique_word_coverage'] = float(np.mean(coverage)) if coverage else None
    verifier = {}
    for label_name, predicate in [('exact_correct', lambda em: em == 1), ('not_exact_correct', lambda em: em == 0)]:
        values = [r['scores']['cross_mean'][i] for r in scored for i in range(5) if r['answers']['view' + str(i)] and predicate(r['candidate_metrics']['view' + str(i)]['em'])]
        verifier[label_name] = {'candidate_count': len(values), 'mean_confidence': float(np.mean(values)) if values else None, 'median_confidence': float(np.median(values)) if values else None}
    candidate_p = [p for r in scored for row in r['matrix'] for p in row]
    verifier.update({'condition_rows': len(candidate_p), 'min_probability': min(candidate_p), 'max_probability': max(candidate_p), 'fraction_p_ge_095': float(np.mean(np.asarray(candidate_p) >= .95)), 'scope': 'Descriptive repeated-candidate diagnostic; not independent calibration or semantic validation.'})
    analysis = {'questions': len(inputs), 'methods': summary, 'comparisons': comparisons, 'primary_contrasts': ['cqc_paper_var__minus__view0', 'cqc_paper_var__minus__maxscore_evidence'], 'selection_disagreements': disagreements, 'paraphrases': {k: sum(r['paraphrase_diagnostics'][k] for r in scored) for k in scored[0]['paraphrase_diagnostics']}, 'formatting_lexical_only': formatting, 'verifier_diagnostics': verifier, 'cache_counts': cache_counts, 'runtime': runtimes, 'bootstrap': {'replicates': 20000, 'seed': config['seed'], 'unit': 'question', 'multiplicity_adjusted': False}, 'scope': config['scope'], 'labels_used_only_after_frozen_selection': True, 'semantic_paraphrase_review': 'Not established by automated checks', 'advance': 'No automatic expansion; review validity and pilot uncertainty first.'}
    return scored, analysis


def report(analysis):
    lines = ['# Stage12 CQC budget-adapted pilot', '', analysis['scope'], '', '## Official answer metrics', '', '| Condition | F1 | EM |', '|---|---:|---:|']
    for key, scores in analysis['methods'].items():
        lines.append(f"| {key} | {scores['f1']:.6f} | {scores['em']:.6f} |")
    lines += ['', '## Prespecified primary contrasts', '']
    for key in analysis['primary_contrasts']:
        metric = analysis['comparisons'][key]['f1']
        lines.append(f"- {key}: F1 difference {metric['mean']:+.6f}, exploratory 95% interval [{metric['ci95'][0]:+.6f}, {metric['ci95'][1]:+.6f}].")
    lines += ['', 'All 45 pairwise comparisons for F1/EM and all 15 rule-choice disagreements are retained in analysis.json. Intervals use 20,000 question-level bootstrap replicates and no multiplicity correction. This repeatedly used 32-question panel and one seed do not establish generalization.', '', '## Implementation validity and cost', '', 'The CPU audit reconstructs all source pools, selection IDs, prompts, cache identities, parsers, rankings, choices and official metrics. Full mode also checks every encoding, packing decision and output-token decoding; replay mode explicitly does not repeat tokenization. Independent GPU checks have a separate artifact and must not be inferred from this report.', '', 'The generated evidence field is reasoning text, not necessarily a quoted supporting span. Lexical coverage/substring diagnostics are not semantic faithfulness. Paraphrase semantic equivalence still requires review.', '', 'All controls share staged caches. Existing dual-dense retrieval/index construction is excluded from measured stage times; these numbers are not uncached end-to-end latency. No training, untouched questions or final submission is part of this experiment.', '', '## Verifier diagnostics', '', json.dumps(analysis['verifier_diagnostics'], indent=2), '', '## Paraphrase parsing diagnostics', '', json.dumps(analysis['paraphrases'], indent=2), '', 'No automatic expansion is authorized by completion of the engineering checks.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['preflight', 'full', 'replay'], required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    config = settings()
    base = ROOT / config['output']
    inputs, manifest = frozen_inputs(config, base)
    from prepare_cqc_budget12 import extract_prompts
    prompt_config = json.loads((ROOT / 'configs/cqc_budget_stage12_prompts.json').read_text(encoding='utf-8'))
    assert extract_prompts() == prompt_config
    tokenizers = load_tokenizers(config) if args.mode != 'replay' else {}
    if args.mode == 'preflight':
        preflight(config, base, inputs, tokenizers, prompt_config)
        return
    out = ROOT / args.output if args.output else base / ('analysis_remote' if args.mode == 'full' else 'analysis_local')
    assert not out.exists(), 'Preserve existing audit output; choose a new directory.'
    decisions, diagnostics, runtimes, cache_counts = check_outputs(config, base, inputs, tokenizers, prompt_config)
    scored, analysis = evaluate(config, base, inputs, decisions, diagnostics, runtimes, cache_counts)
    output_hashes = {path.name: file_sha(path) for path in sorted(base.iterdir()) if path.is_file()}
    out.mkdir(parents=True)
    write_new(out / 'question_pairs.jsonl', scored)
    write_new(out / 'lexical_diagnostics.jsonl', diagnostics)
    atomic_json(out / 'analysis.json', analysis)
    (out / 'report_en.md').write_text(report(analysis), encoding='utf-8')
    audit = {'state': 'passed', 'mode': args.mode, 'tokenization_reconstructed': args.mode == 'full', 'gpu_model_loaded': False, 'questions': len(inputs), 'answer_rows': len(inputs) * 8, 'verifier_rows': len(inputs) * 25, 'cache_counts': cache_counts, 'manifest_sha256': file_sha(base / 'manifest.json'), 'implementation_sha256': file_sha(__file__), 'result_sha256': output_hashes, 'analysis_sha256': file_sha(out / 'analysis.json'), 'official_evaluator_sha256': file_sha(ROOT / 'references/hotpot_eval.txt'), 'labels_sha256': file_sha(ROOT / 'data/stage08_confirmation/labels.jsonl')}
    atomic_json(out / 'audit.json', audit)
    print(json.dumps({'state': 'passed', 'mode': args.mode, 'questions': len(inputs), 'methods': analysis['methods']}), flush=True)


if __name__ == '__main__':
    main()
