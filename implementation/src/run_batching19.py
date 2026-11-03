import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import datetime
import json
import math
import subprocess
import time

import torch
import transformers

from common import ROOT, atomic_json, digest, read_jsonl
from batching19_core import check19, groups, padded_batch
from run_cost17 import request as serial_request
from cqc_budget12_core import answer_messages, file_sha, pack_topk, qwen_prompt
from run_cqc_budget12 import check_pause, load_reader, verify_assets


def gpu_snapshot():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    pids = sorted({int(v.strip()) for v in result.stdout.splitlines() if v.strip()})
    if set(pids) - {os.getpid()}:
        raise RuntimeError('External GPU process detected: ' + str(pids))
    return {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'pids': pids}


def batched_request(row, ranks, arm, cfg, scorer, model, tokenizer):
    before = gpu_snapshot()
    check_pause()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    selected = list(range(len(row['documents'])))
    t1 = time.perf_counter()
    encodings = []
    cuts = []
    for index in selected:
        doc = row['documents'][index]
        old = scorer.truncated
        encodings.append(scorer.encode_one({'question': row['question'], 'passages': [{'title': doc['title'], 'text': doc['text']}]}, (0,)))
        cuts.append(scorer.truncated != old)
    t2 = time.perf_counter()
    plan = groups([len(v) for v in encodings],cfg['batch_size'])
    values = [None]*len(selected)
    layouts = []
    with torch.inference_mode():
        for indices in plan:
            check_pause()
            layout = padded_batch([encodings[i] for i in indices],scorer.tokenizer.pad_token_id)
            batch = {k:torch.tensor(v,dtype=torch.long,device='cuda') for k,v in layout.items()}
            logits = scorer.model(**batch,use_cache=False,logits_to_keep=1).logits[:,-1,:].float()
            scores = (logits[:,scorer.yes]-logits[:,scorer.no]).cpu().tolist()
            for index,value in zip(indices,scores):
                values[index]=float(value)
            layouts.append(layout)
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    assert all(math.isfinite(v) for v in values)
    scores = dict(zip(selected, values))
    order = sorted(selected, key=lambda i: (-scores[i], i))
    packing = pack_topk(row['documents'], order, tokenizer, cfg['context_tokens'], cfg['top_k'])
    t4 = time.perf_counter()
    prompt, prefix = qwen_prompt(tokenizer, answer_messages(row['question'], packing['context'], True))
    assert len(prefix) <= cfg['generator_max_input_tokens']
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to('cuda')
    assert encoded['input_ids'][0].tolist() == prefix
    options = {'do_sample': False, 'max_new_tokens': cfg['max_new_tokens'], 'pad_token_id': tokenizer.eos_token_id, 'repetition_penalty': 1.0}
    torch.cuda.synchronize()
    t5 = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**encoded, **options)
    torch.cuda.synchronize()
    t6 = time.perf_counter()
    ids = output[0, len(prefix):].tolist()
    answer = tokenizer.decode(ids, skip_special_tokens=True).strip()
    t7 = time.perf_counter()
    after = gpu_snapshot()
    return {'id': row['id'], 'arm': arm, 'scored_indices': selected, 'scores': values, 'reranker_inputs': [{'input_ids_sha256': digest(v), 'input_tokens': len(v), 'truncated': cut} for v, cut in zip(encodings, cuts)], 'order': order, 'packing': packing, 'prompt': prompt, 'input_ids_sha256': digest(prefix), 'input_tokens': len(prefix), 'options': options, 'output_ids': ids, 'answer': answer, 'cap_hit': len(ids) == cfg['max_new_tokens'], 'seconds': {'selection': t1-t0, 'reranker_encode': t2-t1, 'reranker_forward': t3-t2, 'ordering_pack': t4-t3, 'reader_prompt': t5-t4, 'reader_generate': t6-t5, 'reader_decode': t7-t6, 'total': t7-t0}, 'gpu_before': before, 'gpu_after': after, 'microbatches': [{'indices':indices,'padded_width':len(layout['input_ids'][0]),'tensor_sha256':{k:digest(v) for k,v in layout.items()}} for indices,layout in zip(plan,layouts)],'forward_invocations':len(plan),'padded_tokens':sum(len(v['input_ids'])*len(v['input_ids'][0]) for v in layouts),'nonpad_tokens':sum(map(len,encodings))}


def request(row,ranks,arm,cfg,scorer,model,tokenizer):
    torch.cuda.reset_peak_memory_stats()
    if arm=='full':
        result=serial_request(row,ranks,'full',cfg,scorer,model,tokenizer)
        total=sum(v['input_tokens'] for v in result['reranker_inputs'])
        result.update(microbatches=None,forward_invocations=len(result['scores']),padded_tokens=total,nonpad_tokens=total)
    else:
        assert arm=='batch4'
        result=batched_request(row,ranks,arm,cfg,scorer,model,tokenizer)
    result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return result


def append(stream, item):
    stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
    stream.flush()


def main():
    from qwen_student import Scorer
    cfg, base, manifest = check19()
    preflight = json.loads((base / 'preflight.json').read_text())
    freeze = file_sha(base / 'manifest.json')
    assert preflight['state'] == 'passed' and preflight['manifest_sha256'] == freeze
    for name in ['requests.jsonl', 'warmup.jsonl', 'runtime.json']:
        assert not (base / name).exists(), 'Preserve interrupted or completed attempts; do not silently resume'
    check_pause()
    gpu_snapshot()
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = read_jsonl(base / 'inputs.jsonl')
    by_id = {r['id']: r for r in rows}
    ranks = {r['id']: {} for r in rows}
    schedule = read_jsonl(base / 'schedule.jsonl')
    state = {'state': 'running', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'pid': os.getpid(), 'manifest_sha256': freeze, 'implementation_sha256': file_sha(__file__), 'scope': cfg['scope'], 'answer_result_cache': False, 'reranker_result_cache': False, 'primary_requests': 0, 'primary_score_calls': 0, 'warmup_requests': 0, 'warmup_score_calls': 0}
    started = time.perf_counter()
    try:
        start = time.perf_counter()
        _, rerank_assets = verify_assets(cfg['reranker'])
        state['reranker_asset_verification_seconds'] = time.perf_counter() - start
        sc = json.loads((ROOT / 'configs/qwen512_v1.json').read_text())
        assert sc['base_model'] == cfg['reranker']['repo'] and sc['base_revision'] == cfg['reranker']['revision'] and sc['max_length'] == cfg['reranker_max_length']
        start = time.perf_counter()
        scorer = Scorer(sc)
        torch.cuda.synchronize()
        state['reranker_load_seconds'] = time.perf_counter() - start
        start = time.perf_counter()
        model, tokenizer, load_seconds, reader_assets = load_reader(cfg['reader'])
        state.update(reader_load_seconds=load_seconds, reader_asset_tokenizer_load_total_seconds=time.perf_counter()-start, reader_assets_digest=reader_assets, reranker_assets_digest=rerank_assets, generation_defaults=model.generation_config.to_dict(), reader_eos_token_id=tokenizer.eos_token_id, torch_version=torch.__version__, transformers_version=transformers.__version__, gpu_name=torch.cuda.get_device_name())
        torch.cuda.reset_peak_memory_stats()
        with (base / 'warmup.jsonl').open('x', encoding='utf-8') as stream:
            for row in rows[:cfg['warmup_questions']]:
                for arm in cfg['arms']:
                    saved = request(row, ranks[row['id']], arm, cfg, scorer, model, tokenizer)
                    append(stream, saved)
                    state['warmup_requests'] += 1
                    state['warmup_score_calls'] += len(saved['scores'])
                    state['peak_allocated_seen'] = max(state.get('peak_allocated_seen',0),saved['peak_allocated_bytes'])
                    state['peak_reserved_seen'] = max(state.get('peak_reserved_seen',0),saved['peak_reserved_bytes'])
        with (base / 'requests.jsonl').open('x', encoding='utf-8') as stream:
            for step in schedule:
                saved = request(by_id[step['id']], ranks[step['id']], step['arm'], cfg, scorer, model, tokenizer)
                append(stream, {**step, **saved})
                state['primary_requests'] += 1
                state['primary_score_calls'] += len(saved['scores'])
                state['peak_allocated_seen'] = max(state.get('peak_allocated_seen',0),saved['peak_allocated_bytes'])
                state['peak_reserved_seen'] = max(state.get('peak_reserved_seen',0),saved['peak_reserved_bytes'])
                if state['primary_requests'] % 8 == 0:
                    print(json.dumps({'requests': state['primary_requests'], 'score_calls': state['primary_score_calls'], 'wall_seconds': time.perf_counter()-started}), flush=True)
        assert state['primary_requests'] == manifest['primary_generation_calls']
        assert state['primary_score_calls'] == manifest['primary_scored_documents']
        state.update(state='complete', completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), gpu_final=gpu_snapshot(), peak_allocated_bytes=state['peak_allocated_seen'], peak_reserved_bytes=state['peak_reserved_seen'], output_sha256={n: file_sha(base/n) for n in ['requests.jsonl','warmup.jsonl']})
    except BaseException as error:
        state.update(state='failed', error_type=type(error).__name__, error=str(error), completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        raise
    finally:
        state['wall_seconds_including_assets_load_warmup_io_monitoring'] = time.perf_counter()-started
        atomic_json(base / 'runtime.json', state)
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
