import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

import contextlib
import datetime
import hashlib
import json
import math
import time

import torch
import transformers
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import AutoTokenizer

from common import ROOT, atomic_json, digest, read_jsonl
from numeric20_core import check20, layouts
from cqc_budget12_core import file_sha, pack_topk
from run_cqc_budget12 import check_pause, verify_assets
from run_cost17 import gpu_snapshot


def canonical(model):
    return {name:hashlib.sha256(p.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest() for name,p in model.named_parameters()}


def backend_flags():
    return {'math':torch.backends.cuda.math_sdp_enabled(),'flash':torch.backends.cuda.flash_sdp_enabled(),'efficient':torch.backends.cuda.mem_efficient_sdp_enabled(),'cudnn':torch.backends.cuda.cudnn_sdp_enabled()}


def request(row,arm,cfg,scorer,tokenizer,encoded):
    check_pause();before=gpu_snapshot();torch.cuda.synchronize();start=time.perf_counter()
    plan=layouts(encoded,arm,scorer.tokenizer.pad_token_id,cfg['batch_size']);values=[None]*len(encoded)
    manager=sdpa_kernel(SDPBackend.MATH) if arm['backend']=='math' else contextlib.nullcontext()
    with manager,torch.inference_mode():
        flags=backend_flags()
        for call in plan:
            check_pause()
            if arm['name']=='bf16_native_implicit':
                found=scorer.score([encoded[call['indices'][0]]]).cpu().tolist()
            else:
                tensors={k:torch.tensor(v,dtype=torch.long,device='cuda') for k,v in call['tensors'].items()}
                logits=scorer.model(**tensors,use_cache=False,logits_to_keep=1).logits[:,-1,:].float()
                found=(logits[:,scorer.yes]-logits[:,scorer.no]).cpu().tolist()
            for index,value in zip(call['indices'],found):values[index]=float(value)
    torch.cuda.synchronize();score_end=time.perf_counter()
    assert all(math.isfinite(v) for v in values)
    order=sorted(range(len(values)),key=lambda i:(-values[i],i));packing=pack_topk(row['documents'],order,tokenizer,cfg['context_tokens'],cfg['top_k'])
    finish=time.perf_counter();after=gpu_snapshot()
    return {'id':row['id'],'arm':arm['name'],'scores':values,'order':order,'packing':packing,'input_evidence':[{'input_ids_sha256':digest(v),'tokens':len(v)} for v in encoded],'calls':[{'indices':v['indices'],'width':len(v['tensors']['input_ids'][0]),'tensor_sha256':{k:digest(t) for k,t in v['tensors'].items()}} for v in plan],'backend_flags':flags,'parameter_dtype':str(next(scorer.model.parameters()).dtype),'seconds_forward_and_layout':score_end-start,'seconds_with_pack':finish-start,'gpu_before':before,'gpu_after':after}


def main():
    from qwen_student import Scorer
    cfg,base,m=check20();pre=json.loads((base/'preflight.json').read_text());assert pre['state']=='passed' and pre['manifest_sha256']==file_sha(base/'manifest.json')
    assert not (base/'requests.jsonl').exists() and not (base/'runtime.json').exists()
    assert torch.__version__==cfg['expected_torch'] and transformers.__version__==cfg['expected_transformers']
    check_pause();gpu_snapshot();torch.manual_seed(cfg['seed']);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.perf_counter();state={'state':'running','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_sha256':file_sha(base/'manifest.json'),'implementation_sha256':file_sha(__file__),'qa_generation_calls':0,'labels_parsed':False,'cpu_encoding_cache':True,'model_result_cache':False,'torch_version':torch.__version__,'transformers_version':transformers.__version__,'gpu':torch.cuda.get_device_name(),'scored_documents':0,'question_conditions':0}
    try:
        _,assets=verify_assets(cfg['reranker']);state['reranker_assets']=assets
        sc=json.loads((ROOT/'configs/qwen512_v1.json').read_text());assert sc['max_length']==cfg['reranker_max_length'] and sc['base_revision']==cfg['reranker']['revision']
        scorer=Scorer(sc)
        reader=AutoTokenizer.from_pretrained(str(ROOT/'models'/cfg['reader']['repo'].replace('/','--')),local_files_only=True)
        rows={r['id']:r for r in read_jsonl(base/'inputs.jsonl')};encoded={}
        for q,row in rows.items():
            encoded[q]=[scorer.encode_one({'question':row['question'],'passages':[{'title':d['title'],'text':d['text']}]},(0,)) for d in row['documents']]
        state['native_backend_flags']=backend_flags()
        state['buffers_before']={n:{'dtype':str(v.dtype),'sha256':hashlib.sha256(v.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()} for n,v in scorer.model.named_buffers()}
        state['weights_before']=canonical(scorer.model);state['parameters']=sum(p.numel() for p in scorer.model.parameters())
        arms={a['name']:a for a in cfg['arms']};converted=False;torch.cuda.reset_peak_memory_stats()
        with (base/'requests.jsonl').open('x',encoding='utf-8') as stream:
            for step in read_jsonl(base/'schedule.jsonl'):
                arm=arms[step['arm']]
                if arm['dtype']=='fp32' and not converted:
                    scorer.model.float();state['weights_after']=canonical(scorer.model);assert state['weights_after']==state['weights_before'];converted=True
                    state['same_parameter_values_verified']=True
                    state['buffers_after']={n:{'dtype':str(v.dtype),'sha256':hashlib.sha256(v.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()} for n,v in scorer.model.named_buffers()}
                assert next(scorer.model.parameters()).dtype==(torch.float32 if arm['dtype']=='fp32' else torch.bfloat16)
                result=request(rows[step['id']],arm,cfg,scorer,reader,encoded[step['id']])
                stream.write(json.dumps({**step,**result},ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                state['scored_documents']+=len(result['scores']);state['question_conditions']+=1
                print(json.dumps({'condition':step['arm'],'phase':step['phase'],'completed':state['question_conditions'],'scored_documents':state['scored_documents'],'seconds':time.perf_counter()-started}),flush=True)
        assert state['scored_documents']==m['primary_scored_documents']+m['warmup_scored_documents'] and state['question_conditions']==81 and converted
        state.update(state='complete',output_sha256=file_sha(base/'requests.jsonl'),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu_final=gpu_snapshot())
    except BaseException as error:
        state.update(state='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        state.update(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.perf_counter()-started);atomic_json(base/'runtime.json',state)
    print(json.dumps({k:v for k,v in state.items() if k not in ['weights_before','weights_after']}),flush=True)


if __name__=='__main__':
    main()
