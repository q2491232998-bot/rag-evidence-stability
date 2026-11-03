import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

import contextlib
import datetime
import hashlib
import json
import time

import torch
import transformers
from torch.nn.attention import sdpa_kernel, SDPBackend

from common import ROOT, atomic_json, read_jsonl
from ipm_crossset_v3_core import check22
from run_cost17 import request as serial_request, gpu_snapshot
from run_batching19 import batched_request
from run_numeric20 import canonical, backend_flags
from cqc_budget12_core import file_sha
from run_cqc_budget12 import check_pause, load_reader, verify_assets


def buffer_hashes(model):
    return {n:{'dtype':str(v.dtype),'sha256':hashlib.sha256(v.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()} for n,v in model.named_buffers()}


class ScopedForward:
    def __init__(self,model):
        self.model=model;self.math=False;self.calls=0;self.last_flags=None
    def __getattr__(self,name):return getattr(self.model,name)
    def __call__(self,*args,**kwargs):
        manager=sdpa_kernel(SDPBackend.MATH) if self.math else contextlib.nullcontext()
        with manager:
            flags=backend_flags()
            if self.math:assert flags=={'math':True,'flash':False,'efficient':False,'cudnn':False}
            self.last_flags=flags;self.calls+=1
            return self.model(*args,**kwargs)


def request(row,ranks,arm,cfg,scorer,model,tokenizer):
    assert arm in cfg['arms'];check_pause();gpu_snapshot()
    torch.cuda.synchronize();begin=time.perf_counter()
    dtype=torch.bfloat16 if arm.startswith('bf16_') else torch.float32
    raw=scorer.model.model
    with torch.no_grad():
        for p in raw.parameters():
            if p.dtype!=dtype:p.data=p.data.to(dtype=dtype)
    assert all(p.dtype==dtype for p in raw.parameters())
    scorer.model.math=arm.startswith('fp32_');scorer.model.calls=0
    torch.cuda.synchronize();conversion=time.perf_counter()-begin
    flags=backend_flags();torch.cuda.reset_peak_memory_stats()
    if arm=='bf16_serial':
        saved=serial_request(row,{},'full',cfg,scorer,model,tokenizer)
        total=sum(v['input_tokens'] for v in saved['reranker_inputs'])
        saved.update(arm=arm,microbatches=None,forward_invocations=len(saved['scores']),padded_tokens=total,nonpad_tokens=total)
    else:
        settings={**cfg,'batch_size':1 if arm=='fp32_serial' else 4}
        saved=batched_request(row,{},arm,settings,scorer,model,tokenizer)
    assert backend_flags()==flags
    assert scorer.model.calls==saved['forward_invocations']
    saved.update(parameter_dtype=str(dtype),reranker_backend_flags=scorer.model.last_flags,reader_native_backend_flags=flags,backend_flags_after=backend_flags(),dtype_transition_seconds_excluded=conversion,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return saved


def append(stream, item):
    stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
    stream.flush()


def main():
    from qwen_student import Scorer
    cfg, base, manifest = check22()
    assert torch.__version__==cfg['expected_torch'] and transformers.__version__==cfg['expected_transformers']
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
        raw_model = scorer.model
        scorer.model = ScopedForward(raw_model)
        initial_weights = canonical(raw_model)
        initial_buffers = buffer_hashes(raw_model)
        state.update(weights_before=initial_weights,buffers_before=initial_buffers,parameter_elements=sum(p.numel() for p in raw_model.parameters()),encoding_cache=False)
        torch.cuda.synchronize()
        state['reranker_load_seconds'] = time.perf_counter() - start
        start = time.perf_counter()
        model, tokenizer, load_seconds, reader_assets = load_reader(cfg['reader'])
        state.update(reader_load_seconds=load_seconds, reader_asset_tokenizer_load_total_seconds=time.perf_counter()-start, reader_assets_digest=reader_assets, reranker_assets_digest=rerank_assets, generation_defaults=model.generation_config.to_dict(), reader_eos_token_id=tokenizer.eos_token_id, torch_version=torch.__version__, transformers_version=transformers.__version__, gpu_name=torch.cuda.get_device_name(),native_backend_flags=backend_flags())
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
                assert time.perf_counter()-started < cfg['maximum_execution_seconds'], 'Frozen time cap'
                assert sum(p.stat().st_size for p in base.rglob('*') if p.is_file()) < cfg['maximum_new_result_bytes'], 'Frozen storage cap'
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
        state.update(weights_after=canonical(raw_model),buffers_after=buffer_hashes(raw_model))
        assert state['weights_before']==state['weights_after'] and state['buffers_before']==state['buffers_after']
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
