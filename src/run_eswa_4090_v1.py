"""Real RTX4090 forwards; secondary calibration is frozen before evaluation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE','1')
import argparse,collections,json,subprocess,time
from eswa_4090_common_v1 import *
import run_eswa_extension_v1 as core
from run_eswa_stage2_v1 import torch_init,parameter_hash,buffer_hashes
core.CFG=C;core.OUT=OUT;core.check_deadline=check_deadline

def hardware():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader'],text=True).strip()
    assert C['gpu_uuid'] in raw and 'RTX 4090' in raw and len(raw.splitlines())==1,raw
    return raw

def epsilon_for(task):
    if task['method']=='band_recalibrated':return load(OUT/'epsilon_recalibrated.json')['models'][task['model']]['epsilon']
    return C['transferred_epsilons'][task['model']]

def request(row,rank,tok,task,monitor):
    internal=dict(task)
    if task['method'].startswith('band_'):internal['method']='band'
    for attempt in range(C['maximum_contaminated_attempts_per_request']):
        check_deadline();monitor.wait();monitor.begin(identity(task))
        try:result=core.request(row,rank,tok,internal,epsilon_for(task))
        except BaseException as e:
            try:observations=monitor.finish_request()
            except BaseException as monitoring_error:observations=[{'monitor_error':repr(monitoring_error)}]
            core.append(OUT/'failed_attempts.jsonl',{**task,'attempt':attempt,'error':repr(e),'utc':utc(),'observations':observations});raise
        contamination=monitor.finish_request();result.update(task)
        result.update(utc=utc(),attempt=attempt,gpu_isolation_observed=not contamination,gpu_sampling_interval_seconds=C['gpu_monitor_interval_seconds'])
        if not contamination:return result
        result['external_gpu_observations']=contamination;core.append(OUT/'contaminated_attempts.jsonl',result)
    raise RuntimeError('Three externally contaminated attempts; stopping without outcome selection')

def execute(model,phase):
    verify();check_deadline();hw=hardware()
    if phase!='calibration':
        e=load(OUT/'epsilon_freeze.json');assert sha(OUT/'epsilon_recalibrated.json')==e['epsilon_sha256']
        for p,h in e['calibration_records_sha256'].items():assert sha(ROOT/p)==h
    monitor=core.Monitor(model+'/'+phase);started=time.monotonic();path=OUT/f'{phase}_{model}.jsonl';assert not path.exists()
    runtime={'state':'running','utc':utc(),'model':model,'phase':phase,'hardware':hw,'run_id':str(time.time_ns())}
    try:
        monitor.wait();torch=torch_init()
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen--Qwen3-8B'),local_files_only=True)
        rank=core.ExtensionRanker(model);before=parameter_hash(rank.model)
        plan=tasks(model,phase);cells=list(dict.fromkeys((t['method'],t['batch_size']) for t in plan));warm=[]
        for row in read_jsonl(DATA/'warmup_inputs.jsonl'):
            for method,size in cells:
                t=dict(model=model,id=row['id'],method=method,batch_size=size,rep=-1,seed=None,role='warmup_'+phase)
                r=request(row,rank,tok,t,monitor);warm.append({k:r[k] for k in ['id','method','batch_size','total_seconds']})
        runtime['warmup']=warm
        rows={r['id']:r for r in read_jsonl(DATA/'cross_hardware_inputs.jsonl' if phase=='cross' else OUT/'inputs.jsonl')}
        for ix,t in enumerate(plan):
            r=request(rows[t['id']],rank,tok,t,monitor);r['run_id']=runtime['run_id'];core.append(path,r)
            if (ix+1)%32==0:print(json.dumps({'phase':phase,'model':model,'completed':ix+1,'expected':len(plan),'seconds':time.monotonic()-started}),flush=True)
        assert parameter_hash(rank.model)==before and buffer_hashes(rank.model)==rank.buffers
        runtime.update(state='complete',records=len(plan),records_sha256=sha(path),parameter_hash=before,parameters_unchanged=True,buffers_unchanged=True,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as e:runtime.update(state='interrupted',error=repr(e));raise
    finally:
        close_error=None
        try:monitor.close()
        except BaseException as e:close_error=e;runtime.update(state='interrupted',monitor_close_error=repr(e))
        runtime.update(finished_utc=utc(),seconds=time.monotonic()-started);save(OUT/f'runtime_{phase}_{model}.json',runtime)
        if close_error:raise close_error

def freeze_epsilon():
    verify();assert not any(OUT.glob('evaluation_*.jsonl')),'Evaluation began before calibration freeze'
    result={'utc':utc(),'pooling':C['epsilon_pooling'],'models':{}};paths={};detail=[]
    for model in ['qwen','bge']:
        path=OUT/f'calibration_{model}.jsonl';info=load(OUT/f'runtime_calibration_{model}.json');assert info['state']=='complete' and sha(path)==info['records_sha256']
        rr=read_jsonl(path);records={identity(r):r for r in rr};plan=tasks(model,'calibration');assert len(rr)==len(plan)==len(records) and set(records)=={identity(t) for t in plan}
        max_error=-1.;witness=None
        for r in rr:
            if r['method']=='serial':continue
            ref=records[model,r['id'],'serial',1,r['rep'],None];assert r['encoded_sha256']==ref['encoded_sha256']
            errors=[abs(b-s) for b,s in zip(r['scores'],ref['scores'])];assert len(errors)==r['candidate_count']==ref['candidate_count']
            imax=max(range(len(errors)),key=lambda i:errors[i]);record={k:r[k] for k in ['model','id','dataset','batch_size','rep','candidate_count']}
            record.update(max_abs_error=errors[imax],candidate_index=imax,batch_score=r['scores'][imax],serial_score=ref['scores'][imax]);detail.append(record)
            if errors[imax]>max_error:max_error=errors[imax];witness=record
        result['models'][model]={'epsilon':max_error*C['epsilon_multiplier'],'maximum_absolute_error':max_error,'witness':witness,'questions':64,'batches':C['batches'],'repetitions':2}
        paths[str(path.relative_to(ROOT))]=sha(path)
    save_rows(OUT/'calibration_errors.jsonl',detail);save(OUT/'epsilon_recalibrated.json',result)
    save(OUT/'epsilon_freeze.json',{'utc':utc(),'epsilon_sha256':sha(OUT/'epsilon_recalibrated.json'),'calibration_records_sha256':paths,'evaluation_not_started':True,'protocol_freeze_sha256':sha(OUT/'freeze.json')})
    print(json.dumps(result),flush=True)

def reader():
    verify();check_deadline();hardware()
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from cqc_budget12_core import qwen_prompt
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen--Qwen3-8B'),local_files_only=True)
    rows={r['id']:r for r in read_jsonl(OUT/'inputs.jsonl')};prompts={};links=[]
    for kind in ['qwen','bge']:
        assert load(OUT/f'runtime_evaluation_{kind}.json')['state']=='complete'
        for r in read_jsonl(OUT/f'evaluation_{kind}.jsonl'):
            messages=[{'role':'system','content':'You are a helpful question answering assistant.'},{'role':'user','content':'Answer the question using the provided passages. Output only a concise answer, with no explanation. If the passages do not provide the answer, output Unanswerable.\n\nPassages:\n'+r['context']+'\n\nQuestion: '+rows[r['id']]['question']}]
            text,ids=qwen_prompt(tok,messages);assert len(ids)<=2048;k=digest({'text':text,'max_new_tokens':64})
            prompts.setdefault(k,{'key':k,'text':text,'input_tokens':len(ids),'input_ids_sha256':digest(ids)})
            links.append({k:r[k] for k in ['model','id','method','batch_size','rep','seed']}|{'key':k})
    save_rows(OUT/'reader_links.jsonl',links);save_rows(OUT/'reader_prompts.jsonl',prompts.values())
    monitor=core.Monitor('reader');started=time.monotonic();runtime={'state':'running','utc':utc()}
    try:
        monitor.wait();torch=torch_init();model=AutoModelForCausalLM.from_pretrained(str(ROOT/'models/Qwen--Qwen3-8B'),torch_dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True).eval()
        options={'do_sample':False,'max_new_tokens':64,'pad_token_id':tok.eos_token_id,'repetition_penalty':1.0}
        def generate(prompt):
            check_deadline();monitor.wait();monitor.begin(prompt['key']);features=tok(prompt['text'],add_special_tokens=False,return_tensors='pt').to('cuda')
            with torch.inference_mode():answer=model.generate(**features,**options)
            ids=answer[0,features.input_ids.shape[1]:].cpu().tolist();observations=monitor.finish_request()
            return {'key':prompt['key'],'output_ids':ids,'answer':tok.decode(ids,skip_special_tokens=True).strip(),'cap_hit':len(ids)==64,'source':'new_4090_generation','gpu_external_observations':observations}
        output={}
        for k,p in prompts.items():
            r=generate(p);core.append(OUT/'answers.jsonl',r);output[k]=r
            if len(output)%32==0:print(json.dumps({'phase':'reader','completed':len(output),'expected':len(prompts),'seconds':time.monotonic()-started}),flush=True)
        repeats=[]
        for k in sorted(prompts)[:16]:
            r=generate(prompts[k]);r['identical']=r['output_ids']==output[k]['output_ids'];repeats.append(r)
        save_rows(OUT/'reader_repeats.jsonl',repeats)
        runtime.update(state='complete',unique_prompts=len(prompts),links=len(links),reused_3090_answers=0,repeated_prompts=len(repeats),repeat_identity=sum(r['identical'] for r in repeats),answers_sha256=sha(OUT/'answers.jsonl'))
    except BaseException as e:runtime.update(state='interrupted',error=repr(e));raise
    finally:
        close_error=None
        try:monitor.close()
        except BaseException as e:close_error=e;runtime.update(state='interrupted',monitor_close_error=repr(e))
        runtime.update(finished_utc=utc(),seconds=time.monotonic()-started);save(OUT/'reader_runtime.json',runtime)
        if close_error:raise close_error

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['calibration','evaluation','cross','freeze_epsilon','reader']);p.add_argument('--model',choices=['qwen','bge']);a=p.parse_args()
    if a.phase in ['calibration','evaluation','cross']:execute(a.model,a.phase)
    else:globals()[a.phase]()
