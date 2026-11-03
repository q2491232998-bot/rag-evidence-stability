import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import datetime
import hashlib
import json
import math
import time

import transformers

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cqc_sampling12 import deterministic_sampling

from common import ROOT, atomic_json, digest, read_jsonl
from cqc_budget12_core import answer_messages, binary_ids, check_freeze, file_sha, pack_topk, parse_answer, parse_paraphrases, prompts, qwen_prompt, settings, verification_prompt, write_new


def verify_assets(details):
    if details['repo'] == 'Qwen/Qwen3-8B':
        record = json.loads((ROOT / 'logs/qwen3_reader_stage10_assets_verified.json').read_text())
    elif details['repo'].startswith('mistralai/'):
        record = json.loads((ROOT / 'logs/second_reader_assets_verified.json').read_text())
    else:
        record = json.loads((ROOT / 'configs/cqc_budget_stage12_reranker_assets.json').read_text())
    assert record['revision'] == details['revision']
    folder = ROOT / 'models' / details['repo'].replace('/', '--')
    for asset in record['files']:
        assert file_sha(folder / asset['name']) == asset['sha256'], asset['name']
    return folder, digest(record)


class Cache:
    def __init__(self, path):
        self.path = path
        rows = read_jsonl(path) if path.exists() else []
        self.rows = {r['key']: r for r in rows}
        assert len(self.rows) == len(rows)
        self.initial = len(rows)
        self.stream = path.open('a', encoding='utf-8')

    def save(self, row):
        assert row['key'] not in self.rows
        self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        self.stream.flush()
        self.rows[row['key']] = row

    def close(self):
        self.stream.close()


def check_pause():
    if (ROOT / 'logs/research.pause').exists():
        raise RuntimeError('Pause marker detected; cache retained')


def load_reader(details):
    folder, assets_hash = verify_assets(details)
    tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
    begin = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(str(folder), torch_dtype=torch.bfloat16, device_map={'': 'cuda'}, attn_implementation='sdpa', local_files_only=True).eval()
    torch.cuda.synchronize()
    return model, tokenizer, time.perf_counter() - begin, assets_hash


def generate(model, tokenizer, messages, cap, sampled, seed, cache, config, freeze):
    text, prefix = qwen_prompt(tokenizer, messages)
    assert len(prefix) <= config['generator_max_input_tokens']
    assert len(prefix) + cap <= model.config.max_position_embeddings
    options = {'do_sample': sampled, 'max_new_tokens': cap, 'pad_token_id': tokenizer.eos_token_id, 'repetition_penalty': 1.0}
    if sampled:
        options.update(temperature=config['paraphrase_temperature'], top_p=config['paraphrase_top_p'], top_k=0)
    key = digest({'prompt': text, 'options': options, 'seed': seed if sampled else None, 'freeze': freeze})
    if key not in cache.rows:
        check_pause()
        if sampled:
            torch.manual_seed(seed)
        encoded = tokenizer(text, add_special_tokens=False, return_tensors='pt').to('cuda')
        torch.cuda.synchronize()
        begin = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**encoded, **deterministic_sampling(options))
        torch.cuda.synchronize()
        ids = output[0, len(prefix):].tolist()
        cache.save({'key': key, 'prompt': text, 'input_ids_sha256': digest(prefix), 'input_tokens': len(prefix), 'output_ids': ids, 'raw': tokenizer.decode(ids, skip_special_tokens=True).strip(), 'cap_hit': len(ids) == cap, 'options': options, 'seed': seed if sampled else None, 'sampling_implementation': 'temperature_then_cpu_top_p_seeded_cuda_multinomial' if sampled else 'greedy', 'seconds': time.perf_counter() - begin})
    return cache.rows[key]


def runtime(base, stage, start, load_seconds, cache, assets, rows, extras=None):
    torch.cuda.synchronize()
    record = {'stage': stage, 'state': 'complete', 'rows': rows, 'new_calls': len(cache.rows) - cache.initial, 'cache_entries': len(cache.rows), 'seconds_excluding_asset_checks_and_load': time.perf_counter() - start, 'load_seconds': load_seconds, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'device': torch.cuda.get_device_name(), 'assets_digest': assets, 'manifest_sha256': file_sha(base / 'manifest.json'), 'implementation_sha256': file_sha(__file__), 'torch_version': torch.__version__, 'transformers_version': transformers.__version__, 'cache_sha256': file_sha(cache.path), 'scope': 'Shared caches and staged execution, excluding existing retrieval/index cost; not isolated uncached end-to-end latency.'}
    record.update(extras or {})
    atomic_json(base / (stage + '_runtime.json'), record)


def paraphrase(config, base, inputs, freeze):
    model, tokenizer, load_seconds, assets = load_reader(config['reader'])
    cache = Cache(base / 'paraphrase_cache.jsonl')
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = []
    for row in inputs:
        seed = int(hashlib.sha256((str(config['seed']) + '|' + row['id']).encode()).hexdigest()[:8], 16)
        messages = [{'role': 'system', 'content': prompts()['paraphrase_system']}, {'role': 'user', 'content': 'Original: ' + row['question']}]
        result = generate(model, tokenizer, messages, config['paraphrase_max_new_tokens'], True, seed, cache, config, freeze)
        paras, flags = parse_paraphrases(result['raw'], row['question'])
        output.append({'id': row['id'], 'original': row['question'], 'queries': [row['question']] + paras, 'key': result['key'], 'raw': result['raw'], 'cap_hit': result['cap_hit'], **flags})
        if len(output) % 4 == 0:
            print(json.dumps({'stage': 'paraphrase', 'questions': len(output), 'seconds': time.perf_counter() - started}), flush=True)
    cache.close()
    write_new(base / 'queries.jsonl', output)
    runtime(base, 'paraphrase', started, load_seconds, cache, assets, len(output), {'generation_defaults': model.generation_config.to_dict(), 'reader_eos_token_id': tokenizer.eos_token_id})


def rerank(config, base, inputs, freeze):
    from qwen_student import Scorer
    _, assets = verify_assets(config['reranker'])
    scorer_config = json.loads((ROOT / 'configs/qwen512_v1.json').read_text())
    assert scorer_config['base_model'] == config['reranker']['repo'] and scorer_config['base_revision'] == config['reranker']['revision']
    begin = time.perf_counter()
    scorer = Scorer(scorer_config)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - begin
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'models' / config['reader']['repo'].replace('/', '--')), local_files_only=True)
    query_map = {r['id']: r for r in read_jsonl(base / 'queries.jsonl')}
    cache = Cache(base / 'reranker_cache.jsonl')
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    results = []
    for row in inputs:
        check_pause()
        views = []
        for j, query in enumerate(query_map[row['id']]['queries']):
            scores = []
            keys = []
            for document in row['documents']:
                blind = {'question': query, 'passages': [{'title': document['title'], 'text': document['text']}]}
                before = scorer.truncated
                ids = scorer.encode_one(blind, (0,))
                truncated = scorer.truncated > before
                key = digest({'input': blind, 'freeze': freeze, 'scorer_config': digest(scorer_config)})
                if key not in cache.rows:
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    with torch.inference_mode():
                        value = float(scorer.score([ids]).item())
                    assert math.isfinite(value)
                    torch.cuda.synchronize()
                    cache.save({'key': key, 'query': query, 'document': blind['passages'][0], 'input_tokens': len(ids), 'input_ids_sha256': digest(ids), 'truncated': truncated, 'score': value, 'seconds': time.perf_counter() - begin})
                assert cache.rows[key]['input_ids_sha256'] == digest(ids)
                scores.append(cache.rows[key]['score'])
                keys.append(key)
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
            packed = pack_topk(row['documents'], order, tokenizer, config['context_tokens'], config['top_k'])
            views.append({'view': j, 'query': query, 'scores': scores, 'score_keys': keys, 'order': order, **packed})
        maximum = [max(view['scores'][i] for view in views) for i in range(len(row['documents']))]
        order = sorted(range(len(maximum)), key=lambda i: (-maximum[i], i))
        combined = dict(pack_topk(row['documents'], order, tokenizer, config['context_tokens'], config['top_k']), order=order, scores=maximum)
        results.append({'id': row['id'], 'views': views, 'maxscore': combined, 'pool_corpus_ids': [d['corpus_id'] for d in row['documents']]})
        print(json.dumps({'stage': 'rerank', 'questions': len(results), 'cache_entries': len(cache.rows), 'seconds': time.perf_counter() - started}), flush=True)
    cache.close()
    write_new(base / 'rankings.jsonl', results)
    runtime(base, 'rerank', started, load_seconds, cache, assets, len(results), {'condition_pairs': sum(5 * len(r['documents']) for r in inputs), 'truncated_condition_inputs': scorer.truncated})


def answers(config, base, inputs, freeze):
    model, tokenizer, load_seconds, assets = load_reader(config['reader'])
    ranking = {r['id']: r for r in read_jsonl(base / 'rankings.jsonl')}
    cache = Cache(base / 'answer_cache.jsonl')
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = []
    for row in inputs:
        rank = ranking[row['id']]
        requests = [(f'view{v["view"]}', v['query'], v['context'], False) for v in rank['views']]
        requests += [('original_neutral', row['question'], rank['views'][0]['context'], True), ('maxscore_neutral', row['question'], rank['maxscore']['context'], True), ('maxscore_evidence', row['question'], rank['maxscore']['context'], False)]
        for condition, query, context, neutral in requests:
            assert len(tokenizer.encode(context, add_special_tokens=False)) <= config['context_tokens']
            cap = config['neutral_max_new_tokens'] if neutral else config['answer_max_new_tokens']
            saved = generate(model, tokenizer, answer_messages(query, context, neutral), cap, False, config['seed'], cache, config, freeze)
            parsed = {'answer': saved['raw'], 'evidence': '', 'format_has_evidence': False} if neutral else parse_answer(saved['raw'])
            output.append({'id': row['id'], 'condition': condition, 'query': query, 'context_sha256': digest(context), 'key': saved['key'], 'raw': saved['raw'], 'cap_hit': saved['cap_hit'], **parsed})
        print(json.dumps({'stage': 'answers', 'questions': len(output) // 8, 'cache_entries': len(cache.rows), 'seconds': time.perf_counter() - started}), flush=True)
    cache.close()
    write_new(base / 'answers.jsonl', output)
    runtime(base, 'answers', started, load_seconds, cache, assets, len(output), {'generation_defaults': model.generation_config.to_dict(), 'reader_eos_token_id': tokenizer.eos_token_id})


def verify(config, base, inputs, freeze):
    model, tokenizer, load_seconds, assets = load_reader(config['evaluator'])
    token_ids = binary_ids(tokenizer)
    yes, no = token_ids['Yes']['chosen'], token_ids['No']['chosen']
    query_map = {r['id']: r for r in read_jsonl(base / 'queries.jsonl')}
    candidates = {(r['id'], r['condition']): r for r in read_jsonl(base / 'answers.jsonl')}
    cache = Cache(base / 'verifier_cache.jsonl')
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = []
    for row in inputs:
        check_pause()
        for i in range(5):
            candidate = candidates[row['id'], 'view' + str(i)]
            for j, query in enumerate(query_map[row['id']]['queries']):
                prompt = verification_prompt(query, candidate['answer'], candidate['evidence'])
                ids = tokenizer(prompt, add_special_tokens=True).input_ids
                assert len(ids) <= config['verifier_max_input_tokens']
                assert ids[0] == tokenizer.bos_token_id and ids[1] != tokenizer.bos_token_id
                key = digest({'prompt': prompt, 'yes': yes, 'no': no, 'freeze': freeze})
                if key not in cache.rows:
                    encoded = tokenizer(prompt, add_special_tokens=True, return_tensors='pt').to('cuda')
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    with torch.inference_mode():
                        logits = model(**encoded, use_cache=False, logits_to_keep=1).logits[0, -1, [yes, no]].float()
                        probability = float(logits.softmax(-1)[0].item())
                    torch.cuda.synchronize()
                    cache.save({'key': key, 'prompt': prompt, 'input_tokens': len(ids), 'input_ids_sha256': digest(ids), 'yes_logit': float(logits[0].item()), 'no_logit': float(logits[1].item()), 'probability': probability, 'seconds': time.perf_counter() - begin})
                saved = cache.rows[key]
                output.append({'id': row['id'], 'candidate_view': i, 'query_view': j, 'key': key, 'probability': saved['probability']})
        print(json.dumps({'stage': 'verify', 'questions': len(output) // 25, 'cache_entries': len(cache.rows), 'seconds': time.perf_counter() - started}), flush=True)
    cache.close()
    write_new(base / 'verification.jsonl', output)
    runtime(base, 'verify', started, load_seconds, cache, assets, len(output), {'binary_token_ids': token_ids, 'serialization': 'Raw prompt; tokenizer default single BOS; no chat template; no truncation'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['paraphrase', 'rerank', 'answers', 'verify'], required=True)
    args = parser.parse_args()
    config = settings()
    base = ROOT / config['output']
    check_freeze(base)
    preflight = json.loads((base / 'preflight.json').read_text())
    assert preflight['state'] == 'passed' and preflight['manifest_sha256'] == file_sha(base / 'manifest.json')
    check_pause()
    targets = {'paraphrase': 'queries.jsonl', 'rerank': 'rankings.jsonl', 'answers': 'answers.jsonl', 'verify': 'verification.jsonl'}
    if (base / targets[args.stage]).exists():
        previous = json.loads((base / (args.stage + '_runtime.json')).read_text())
        assert previous['state'] == 'complete' and previous['manifest_sha256'] == file_sha(base / 'manifest.json')
        print(json.dumps({'stage': args.stage, 'state': 'already_complete_no_rerun'}), flush=True)
        return
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = read_jsonl(base / 'inputs.jsonl')
    assert len(rows) == config['questions'] and all(set(r) == {'id', 'question', 'documents', 'source_document_indices'} for r in rows)
    print(json.dumps({'stage': args.stage, 'event': 'start', 'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}), flush=True)
    {'paraphrase': paraphrase, 'rerank': rerank, 'answers': answers, 'verify': verify}[args.stage](config, base, rows, file_sha(base / 'manifest.json'))


if __name__ == '__main__':
    main()
