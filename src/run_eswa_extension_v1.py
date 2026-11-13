"""Actual extension forwards. Selection never reads a complete reference score vector."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import contextlib
import hashlib
import json
import math
import random
import subprocess
import threading
import time
from eswa_extension_common_v1 import *
from run_eswa_stage2_v1 import Ranker, Packer, torch_init, buffer_hashes, parameter_hash, verify_assets
from eswa_gate_methods_v2 import order, flagged

def append(path,record):
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');f.flush()

def snapshot():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True)
    procs=[]
    for line in raw.splitlines():
        if not line.strip():continue
        pid,name,memory=line.split(',',2);pid=int(pid.strip())
        args=subprocess.run(['ps','-p',str(pid),'-o','args='],capture_output=True,text=True).stdout
        procs.append({'pid':pid,'executable':name.strip(),'memory_mib':memory.strip(),'own':pid==os.getpid(),
                      'speech_service_detected':any(s in args.lower() for s in ['local_tts_server','kokoro','tts_server'])})
    return procs

class Monitor:
    def __init__(self,label):
        self.label=label;self.path=OUT/'gpu_occupancy.jsonl';self.stop=threading.Event();self.lock=threading.RLock()
        self.current=None;self.foreign=[];self.error=None;self.last=None
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def observe(self):
        procs=snapshot();foreign=[p for p in procs if not p['own']]
        with self.lock:
            marker=(self.current,tuple((p['pid'],p['own'],p['speech_service_detected']) for p in procs))
            if marker!=self.last:
                append(self.path,{'utc':utc(),'label':self.label,'request':self.current,'processes':procs});self.last=marker
            if self.current and foreign:self.foreign.append({'utc':utc(),'processes':foreign})
        return foreign
    def loop(self):
        while not self.stop.wait(CFG['gpu_monitor_interval_seconds']):
            try:self.observe()
            except BaseException as e:self.error=repr(e);return
    def wait(self):
        self.end()
        while self.observe():
            check_deadline();time.sleep(5)
        assert self.error is None,self.error
    def begin(self,ident):
        with self.lock:self.current=str(ident);self.foreign=[]
        self.observe()
    def end(self):
        with self.lock:
            observations=list(self.foreign);self.current=None
        return observations
    def finish_request(self):
        self.observe();assert self.error is None,self.error
        return self.end()
    def close(self):
        self.stop.set();self.thread.join(timeout=5);assert self.error is None,self.error

def check_deadline():
    check()
    assert time.time()<load(OUT/'queue_started.json')['deadline_epoch'],'Supplement window expired'
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<CFG['maximum_result_bytes'],'Supplement result cap'

class ExtensionRanker(Ranker):
    def score_ordered(self,encoded,indices,size,math_backend=False,preserve=False):
        if not preserve:return self.score_indices(encoded,indices,size,math_backend)
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from batching19_core import padded_batch
        scores={};forward=0.;calls=0
        manager=sdpa_kernel(SDPBackend.MATH) if math_backend else contextlib.nullcontext()
        with torch.inference_mode(),manager:
            for start in range(0,len(indices),size):
                selected=indices[start:start+size]
                assert self.kind=='qwen','Only Qwen random grouping is authorized'
                raw=padded_batch([encoded[i]['input_ids'] for i in selected],self.tok.pad_token_id)
                features={k:torch.tensor(v,device='cuda') for k,v in raw.items()}
                torch.cuda.synchronize();begin=time.perf_counter()
                logits=self.model(**features,use_cache=False,logits_to_keep=1).logits[:,-1,:].float()
                values=(logits[:,self.scorer.yes]-logits[:,self.scorer.no]).cpu().tolist()
                torch.cuda.synchronize();forward+=time.perf_counter()-begin;calls+=1
                assert all(math.isfinite(x) for x in values);scores.update(zip(selected,values))
        assert set(scores)==set(indices)
        return scores,forward,calls

def permutation(qid,n,seed):
    indices=list(range(n))
    if seed is not None:
        value=int(hashlib.sha256(f'eswa-extension|{seed}|{qid}'.encode()).hexdigest(),16)
        random.Random(value).shuffle(indices)
    return indices

def request(row,rank,tok,task,epsilon):
    import torch
    method=task['method'];fp32=method=='fp32_math_serial';tau=CFG['taus'][rank.kind]
    rank.precision(fp32);torch.cuda.synchronize();begin=time.perf_counter()
    encoded,truncated=rank.encode(row);packer=Packer(row['documents'],tok)
    perm=permutation(row['id'],len(encoded),task['seed'])
    initial,forward,calls=rank.score_ordered(encoded,perm,task['batch_size'],fp32,task['seed'] is not None)
    scores=[initial[i] for i in range(len(encoded))];initial_scores=list(scores)
    marked=set();rounds=[];gate_time=0.;kth=None;band=None
    if method in ['gate_v2','random_gate_v2']:
        while True:
            start=time.perf_counter();fresh,pairs=flagged(scores,marked,packer,tau,'v2');gate_time+=time.perf_counter()-start
            if not fresh:break
            actual,seconds,ncalls=rank.score_indices(encoded,fresh,1,False);forward+=seconds;calls+=ncalls
            rounds.append({'indices':fresh,'actual_serial_scores':actual,'trigger_pairs':pairs})
            for i,v in actual.items():scores[i]=v
            marked.update(fresh);assert len(rounds)<=len(scores)
    elif method in ['top15','band']:
        start=time.perf_counter()
        if method=='band':fresh,kth=band_indices(scores,epsilon);band=list(fresh)
        else:fresh=order(scores)[:15]
        gate_time+=time.perf_counter()-start
        actual,seconds,ncalls=rank.score_indices(encoded,fresh,1,False);forward+=seconds;calls+=ncalls
        rounds.append({'indices':fresh,'actual_serial_scores':actual})
        for i,v in actual.items():scores[i]=v
        marked.update(fresh)
    final_order=sorted(band,key=lambda i:(-scores[i],i))[:10] if band is not None else order(scores)
    packed=packer(final_order);torch.cuda.synchronize();total=time.perf_counter()-begin
    assert buffer_hashes(rank.model)==rank.buffers
    return {**task,'dataset':row['dataset'],'paper_id':row.get('paper_id',''),'tau':tau if 'gate' in method else None,
        'epsilon':epsilon if method=='band' else None,'batch_kth_score':kth,'band_indices':band,
        'candidate_count':len(scores),'initial_scores':initial_scores,'scores':scores,'order':final_order,**packed,
        'rescored_indices':sorted(marked),'rescored_count':len(marked),'rescored_fraction':len(marked)/len(scores),
        'rounds':rounds,'total_seconds':total,'forward_seconds':forward,'gate_check_seconds':gate_time,'forward_calls':calls,
        'pack_requests':packer.calls,'pack_cache_misses':packer.misses,'encoded_sha256':digest(encoded),
        'input_tokens':[len(x['input_ids']) for x in encoded],'truncated':truncated,
        'grouping_permutation':perm if task['seed'] is not None else None}

def safe_request(row,rank,tok,task,epsilon,monitor):
    # Invalid attempts are selected only by observed external processes, never by outcomes.
    for attempt in range(CFG['maximum_contaminated_attempts_per_request']):
        check_deadline();monitor.wait();monitor.begin(identity(task));result=None
        try:result=request(row,rank,tok,task,epsilon)
        except BaseException:
            monitor.end();raise
        contamination=monitor.finish_request()
        result.update(utc=utc(),attempt=attempt,gpu_isolation_observed=not contamination,gpu_sampling_interval_seconds=.5)
        if not contamination:return result
        result['external_gpu_observations']=contamination
        append(OUT/'contaminated_attempts.jsonl',result)
    raise RuntimeError('Three externally contaminated attempts; stopped without selecting by result')

def execute(kind):
    verify();verify_assets();torch=torch_init()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen--Qwen3-8B'),local_files_only=True)
    rank=ExtensionRanker(kind);before=parameter_hash(rank.model);epsilon=load(OUT/'epsilon.json')['models'][kind]['epsilon']
    monitor=Monitor(kind);run_id=str(time.time_ns());began=time.monotonic();path=OUT/f'requests_{kind}.jsonl'
    assert not path.exists(),'Use a separately documented continuation for partial executions'
    runtime={'utc':utc(),'state':'running','model':kind,'run_id':run_id,'epsilon':epsilon,'tau':CFG['taus'][kind]}
    try:
        old=[r for r in read_jsonl(ROOT/'results/eswa_supplement_v2/inputs.jsonl') if r['dataset']=='qasper'][:2]
        warm=[]
        cells=list(dict.fromkeys((t['method'],t['batch_size'],t['seed']) for t in tasks(kind)))
        for row in old:
            for method,size,seed in cells:
                t=dict(model=kind,id=row['id'],method=method,batch_size=size,rep=-1,seed=seed,role='warmup')
                out=safe_request(row,rank,tok,t,epsilon,monitor);warm.append({k:out[k] for k in ['id','method','batch_size','seed','total_seconds']})
        runtime['warmup']=warm
        rows={r['id']:r for r in read_jsonl(STAGE2/'inputs.jsonl')};planned=tasks(kind)
        for ix,task in enumerate(planned):
            result=safe_request(rows[task['id']],rank,tok,task,epsilon,monitor);result['run_id']=run_id;append(path,result)
            if (ix+1)%32==0:print(json.dumps({'model':kind,'completed':ix+1,'expected':len(planned),'seconds':time.monotonic()-began}),flush=True)
        assert parameter_hash(rank.model)==before and buffer_hashes(rank.model)==rank.buffers
        runtime.update(state='complete',records=len(planned),records_sha256=sha(path),parameter_hash=before,parameters_unchanged=True,buffers_unchanged=True,
                       gpu=torch.cuda.get_device_name(0),peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as e:runtime.update(state='interrupted',error=repr(e));raise
    finally:
        monitor.close();runtime.update(finished_utc=utc(),seconds=time.monotonic()-began);save(OUT/f'runtime_{kind}.json',runtime)

def reader():
    verify();verify_assets();torch=torch_init()
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from cqc_budget12_core import qwen_prompt
    folder=str(ROOT/'models/Qwen--Qwen3-8B');tok=AutoTokenizer.from_pretrained(folder,local_files_only=True)
    rows={r['id']:r for r in read_jsonl(STAGE2/'inputs.jsonl')};prompts={};links=[]
    for kind in ['qwen','bge']:
        assert load(OUT/f'runtime_{kind}.json')['state']=='complete'
        for r in read_jsonl(OUT/f'requests_{kind}.jsonl'):
            if r['role']=='random':continue
            messages=[{'role':'system','content':'You are a helpful question answering assistant.'},
                      {'role':'user','content':'Answer the question using the provided passages. Output only a concise answer, with no explanation. If the passages do not provide the answer, output Unanswerable.\n\nPassages:\n'+r['context']+'\n\nQuestion: '+rows[r['id']]['question']}]
            text,ids=qwen_prompt(tok,messages);assert len(ids)<=2048;key=digest({'text':text,'max_new_tokens':64})
            prompts.setdefault(key,{'key':key,'text':text,'input_tokens':len(ids),'input_ids_sha256':digest(ids)})
            links.append({k:r[k] for k in ['model','id','method','batch_size','rep','seed']}|{'key':key})
    save_rows(OUT/'reader_links.jsonl',links);save_rows(OUT/'reader_prompts.jsonl',prompts.values())
    old={r['key']:r for r in read_jsonl(STAGE2/'answers.jsonl')};oldprompts={r['key']:r for r in read_jsonl(STAGE2/'reader_prompts.jsonl')}
    output={};new=0;monitor=Monitor('reader');start=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(folder,torch_dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True).eval()
    options={'do_sample':False,'max_new_tokens':64,'pad_token_id':tok.eos_token_id,'repetition_penalty':1.0}
    def generate(prompt):
        check_deadline();monitor.wait();monitor.begin(prompt['key'])
        features=tok(prompt['text'],add_special_tokens=False,return_tensors='pt').to('cuda')
        with torch.inference_mode():answer=model.generate(**features,**options)
        ids=answer[0,features.input_ids.shape[1]:].cpu().tolist();observations=monitor.finish_request()
        # Reader latency is not an outcome; disclose occupation without suppressing output.
        return {'key':prompt['key'],'output_ids':ids,'answer':tok.decode(ids,skip_special_tokens=True).strip(),'cap_hit':len(ids)==64,
                'source':'new_generation','gpu_external_observations':observations}
    try:
        for key,prompt in prompts.items():
            if key in old:
                assert prompt['text']==oldprompts[key]['text'] and prompt['input_ids_sha256']==oldprompts[key]['input_ids_sha256']
                record=dict(old[key],source='stage2_exact_prompt_cache')
            else:record=generate(prompt);new+=1
            append(OUT/'answers.jsonl',record);output[key]=record
            if len(output)%32==0:print(json.dumps({'phase':'reader','completed':len(output),'unique':len(prompts),'new':new}),flush=True)
        repeated=[]
        for key in sorted(prompts)[:16]:
            r=generate(prompts[key]);r['identical']=r['output_ids']==output[key]['output_ids'];repeated.append(r)
        save_rows(OUT/'reader_repeats.jsonl',repeated)
        save(OUT/'reader_runtime.json',{'state':'complete','utc':utc(),'unique_prompts':len(prompts),'new_prompts':new,'reused_stage2':len(prompts)-new,
              'links':len(links),'repeated_prompts':len(repeated),'repeat_identity':sum(r['identical'] for r in repeated),
              'answers_sha256':sha(OUT/'answers.jsonl'),'seconds':time.monotonic()-start})
    finally:monitor.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['execute','reader']);p.add_argument('--model',choices=['qwen','bge']);args=p.parse_args()
    if args.phase=='execute':execute(args.model)
    else:reader()
