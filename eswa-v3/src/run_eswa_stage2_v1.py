"""Actual staged GPU evaluation; gates never access complete reference serial scores."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import contextlib
import gc
import hashlib
import json
import math
import random
import subprocess
import time
from eswa_stage2_common import *
from eswa_gate_methods_v2 import order,flagged
from cqc_budget12_core import qwen_prompt


def gpu_check():
    check()
    pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    assert all(int(x)==os.getpid() for x in pids.split() if x.isdigit()),'Another GPU process appeared'


def torch_init():
    import torch
    import numpy as np
    torch.set_num_threads(4);torch.manual_seed(CFG['seed']);np.random.seed(CFG['seed']);random.seed(CFG['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    assert torch.__version__=='2.5.1+cu121'
    return torch


def buffer_hashes(model):
    import torch
    return {n:{'dtype':str(v.dtype),'sha256':hashlib.sha256(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
            for n,v in model.named_buffers()}


def parameter_hash(model):
    h=hashlib.sha256()
    for name,value in model.named_parameters():h.update(name.encode());h.update(value.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


class Packer:
    def __init__(self,documents,tokenizer):
        self.blocks=['Title: '+d['title']+'\n'+d['text'] for d in documents]
        self.tokenizer=tokenizer;self.cache={};self.calls=0;self.misses=0
    def __call__(self,ranking):
        self.calls+=1;key_=tuple(ranking[:CFG['top_k']])
        if key_ in self.cache:return self.cache[key_]
        self.misses+=1;selected=[];text=''
        for i in key_:
            proposal='\n\n'.join(self.blocks[j] for j in selected+[i])
            if len(self.tokenizer.encode(proposal,add_special_tokens=False))<=CFG['context_tokens']:
                selected.append(i);text=proposal
        result={'selected':selected,'context':text,'tokens':len(self.tokenizer.encode(text,add_special_tokens=False))}
        self.cache[key_]=result;return result


class Ranker:
    def __init__(self,kind):
        import torch
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        self.kind=kind
        if kind=='qwen':
            from qwen_student import Scorer
            cfg=load(ROOT/'configs/qwen512_v1.json');cfg['max_length']=CFG['reranker_max_length']
            assert cfg['base_model']==CFG['qwen']['repo']
            self.scorer=Scorer(cfg);self.model=self.scorer.model;self.tok=self.scorer.tokenizer
        else:
            folder=str(ROOT/'models/BAAI--bge-reranker-v2-m3')
            self.tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,padding_side='right')
            self.model=AutoModelForSequenceClassification.from_pretrained(folder,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
        self.buffers=buffer_hashes(self.model)
    def precision(self,fp32):
        import torch
        dtype=torch.float32 if fp32 else torch.bfloat16
        with torch.no_grad():
            for p in self.model.parameters():
                if p.dtype!=dtype:p.data=p.data.to(dtype)
        assert buffer_hashes(self.model)==self.buffers
    def encode(self,row):
        encoded=[];truncated=[]
        for d in row['documents']:
            if self.kind=='qwen':
                before=self.scorer.truncated
                ids=self.scorer.encode_one({'question':row['question'],'passages':[d]},(0,))
                encoded.append({'input_ids':ids});truncated.append(self.scorer.truncated>before)
            else:
                text='Title: '+d['title']+'\n'+d['text']
                full=self.tok(row['question'],text,truncation=False)
                truncated.append(len(full['input_ids'])>CFG['reranker_max_length'])
                encoded.append(dict(self.tok(row['question'],text,truncation=True,max_length=CFG['reranker_max_length'])))
        return encoded,truncated
    def score_indices(self,encoded,indices,size,math_backend=False):
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from batching19_core import padded_batch
        indices=sorted(indices,key=lambda i:(len(encoded[i]['input_ids']),i))
        scores={};forward=0.;calls=0
        manager=sdpa_kernel(SDPBackend.MATH) if math_backend else contextlib.nullcontext()
        with torch.inference_mode(),manager:
            for start in range(0,len(indices),size):
                selected=indices[start:start+size]
                if self.kind=='qwen':
                    raw=padded_batch([encoded[i]['input_ids'] for i in selected],self.tok.pad_token_id)
                    features={k:torch.tensor(v,device='cuda') for k,v in raw.items()}
                else:features=self.tok.pad([encoded[i] for i in selected],padding=True,return_tensors='pt').to('cuda')
                torch.cuda.synchronize();begin=time.perf_counter()
                if self.kind=='qwen':
                    logits=self.model(**features,use_cache=False,logits_to_keep=1).logits[:,-1,:].float()
                    values=logits[:,self.scorer.yes]-logits[:,self.scorer.no]
                else:values=self.model(**features).logits[:,0].float()
                values=values.cpu().tolist();torch.cuda.synchronize();forward+=time.perf_counter()-begin;calls+=1
                assert all(math.isfinite(x) for x in values)
                scores.update(zip(selected,values))
        assert set(scores)==set(indices)
        return scores,forward,calls


def request(row,ranker,reader_tok,method,batch_size,tau):
    import torch
    gpu_check();ranker.precision(method=='fp32_math_batch4');torch.cuda.synchronize()
    start=time.perf_counter();encoded,truncated=ranker.encode(row)
    packer=Packer(row['documents'],reader_tok)
    initial,forward,calls=ranker.score_indices(encoded,list(range(len(encoded))),batch_size,method=='fp32_math_batch4')
    scores=[initial[i] for i in range(len(encoded))];initial_values=list(scores)
    marked=set();rounds=[];gate_seconds=0.
    if method in ['gate_v1','gate_v2']:
        while True:
            begin=time.perf_counter();fresh,triggers=flagged(scores,marked,packer,tau,'v1' if method=='gate_v1' else 'v2');gate_seconds+=time.perf_counter()-begin
            if not fresh:break
            assert not set(fresh)&marked
            actual,seconds,ncalls=ranker.score_indices(encoded,fresh,1,False)
            forward+=seconds;calls+=ncalls
            rounds.append({'indices':fresh,'actual_serial_scores':actual,'trigger_pairs':triggers})
            for i,value in actual.items():scores[i]=value
            marked.update(fresh);assert len(rounds)<=len(scores)
    elif method=='top15':
        fresh=order(scores)[:15];actual,seconds,ncalls=ranker.score_indices(encoded,fresh,1,False)
        forward+=seconds;calls+=ncalls
        for i,value in actual.items():scores[i]=value
        marked.update(fresh);rounds.append({'indices':fresh,'actual_serial_scores':actual})
    final_order=order(scores);packed=packer(final_order)
    torch.cuda.synchronize();total=time.perf_counter()-start
    assert buffer_hashes(ranker.model)==ranker.buffers
    return {'id':row['id'],'dataset':row['dataset'],'paper_id':row.get('paper_id',''),'model':ranker.kind,'method':method,
            'batch_size':batch_size,'tau':tau if method.startswith('gate_') else None,
            'candidate_count':len(scores),'initial_scores':initial_values,'scores':scores,'order':final_order,**packed,
            'rescored_indices':sorted(marked),'rescored_count':len(marked),'rescored_fraction':len(marked)/len(scores),
            'rounds':rounds,'total_seconds':total,'forward_seconds':forward,'gate_check_seconds':gate_seconds,
            'forward_calls':calls,'pack_requests':packer.calls,'pack_cache_misses':packer.misses,
            'encoded_sha256':digest(encoded),'input_tokens':[len(x['input_ids']) for x in encoded],'truncated':truncated}


def verify_freeze():
    freeze=load(BASE/'freeze.json')
    for path,want in freeze['files_sha256'].items():assert sha(ROOT/path)==want,path
    return freeze


def verify_assets():
    for path,want in load(BASE/'freeze.json')['assets_sha256'].items():assert sha(ROOT/path)==want,path


def warm(rank,tok,tau,all_methods=True):
    old=[r for r in read_jsonl(ROOT/'results/eswa_supplement_v2/inputs.jsonl') if r['dataset']=='qasper'][:2]
    outputs=[]
    for row in old:
        for method in CFG['methods'] if all_methods else ['serial','batch4']:
            result=request(row,rank,tok,method,1 if method=='serial' else 4,tau)
            outputs.append({'id':row['id'],'method':method,'seconds':result['total_seconds']})
    return outputs


def calibration():
    torch=torch_init();gpu_check();verify_freeze();verify_assets()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen--Qwen3-8B'),local_files_only=True)
    rank=Ranker('bge');before=parameter_hash(rank.model);warmup=warm(rank,tok,.03125,False)
    selected=[r for r in read_jsonl(BASE/'inputs.jsonl') if r['role']=='bge_calibration'];assert len(selected)==64
    path=BASE/'bge_calibration_requests.jsonl';assert not path.exists()
    stored={}
    with path.open('x',encoding='utf-8') as f:
        for i,row in enumerate(selected):
            for method in ['serial','batch4'] if i%2==0 else ['batch4','serial']:
                result=request(row,rank,tok,method,1 if method=='serial' else 4,None);result['rep']=0
                stored[row['id'],method]=result;f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush()
            if (i+1)%8==0:print(json.dumps({'phase':'bge_calibration','questions':i+1}),flush=True)
    assert parameter_hash(rank.model)==before
    grid=[];baseline=0
    for row in selected:
        baseline+=stored[row['id'],'serial']['context']!=stored[row['id'],'batch4']['context']
    for tau in CFG['tau_grid']:
        diffs=0;fractions=[]
        for row in selected:
            serial=stored[row['id'],'serial'];batch=stored[row['id'],'batch4'];scores=list(batch['scores']);marked=set();packer=Packer(row['documents'],tok)
            while True:
                fresh,_=flagged(scores,marked,packer,tau,'v1')
                if not fresh:break
                for idx in fresh:scores[idx]=serial['scores'][idx]
                marked.update(fresh)
            diffs+=packer(order(scores))['context']!=serial['context'];fractions.append(len(marked)/len(scores))
        grid.append({'tau':tau,'context_differences':diffs,'baseline_context_differences':baseline,
                     'mean_rescored_fraction':sum(fractions)/len(fractions),'max_rescored_fraction':max(fractions),
                     'joint_feasible':bool(baseline and diffs*10<=baseline and sum(fractions)/len(fractions)<=.2)})
    passing=[r for r in grid if r['joint_feasible']];budget=[r for r in grid if r['mean_rescored_fraction']<=.2]
    if baseline==0:choice=grid[0];rule='no_baseline_events_smallest_tau_unidentified'
    elif passing:choice=min(passing,key=lambda r:(r['mean_rescored_fraction'],r['context_differences'],r['tau']));rule='minimum_cost_feasible'
    elif budget:choice=min(budget,key=lambda r:(r['context_differences'],r['mean_rescored_fraction'],r['tau']));rule='best_within_budget_no_joint_feasible'
    else:choice=min(grid,key=lambda r:(r['mean_rescored_fraction'],r['context_differences'],r['tau']));rule='no_budget_feasible'
    save(BASE/'bge_tau_selection.json',{'utc':utc(),'tau':choice['tau'],'rule':rule,'grid':grid,'calibration_n':64,
          'evaluation_scored':False,'v2_inherits_same_tau':True,'weights_unchanged':True,'parameter_hash':before,'warmup':warmup,
          'calibration_sha256':sha(path),'freeze_sha256':sha(BASE/'freeze.json')})
    print(json.dumps({'state':'bge_tau_frozen','tau':choice['tau'],'rule':rule,'grid':grid}),flush=True)


def schedule(kind):
    inputs=[r for r in read_jsonl(BASE/'inputs.jsonl') if r['role']=='evaluation'];tasks=[]
    for rep in range(2):
        for idx,row in (list(enumerate(inputs)) if rep==0 else list(reversed(list(enumerate(inputs))))):
            methods=CFG['methods'][idx%6:]+CFG['methods'][:idx%6]
            if rep:methods=list(reversed(methods))
            for method in methods:tasks.append({'id':row['id'],'rep':rep,'method':method,'batch_size':1 if method=='serial' else 4,'role':'main'})
    if kind=='qwen':
        qa=[r for r in inputs if r['dataset']=='qasper']
        for rep in range(2):
            for idx,row in enumerate(qa):
                sizes=[2,8,16,32];sizes=sizes[idx%4:]+sizes[:idx%4]
                if rep:sizes=list(reversed(sizes))
                for size in sizes:tasks.append({'id':row['id'],'rep':rep,'method':'batch_scan','batch_size':size,'role':'scan'})
    return tasks


def execute(kind,resume=False):
    torch=torch_init();gpu_check();freeze=verify_freeze();verify_assets()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen--Qwen3-8B'),local_files_only=True)
    tau=CFG['qwen_tau'] if kind=='qwen' else load(BASE/'bge_tau_selection.json')['tau']
    # Require calibration to complete before either model touches new evaluation questions.
    assert (BASE/'bge_tau_selection.json').exists()
    path=BASE/f'requests_{kind}.jsonl';assert resume or not path.exists()
    prior=read_jsonl(path) if path.exists() else []
    keys={(r['id'],r['rep'],r['method'],r['batch_size']) for r in prior};assert len(keys)==len(prior)
    tasks=schedule(kind);expected={(r['id'],r['rep'],r['method'],r['batch_size']) for r in tasks};assert keys<=expected
    rows={r['id']:r for r in read_jsonl(BASE/'inputs.jsonl')};rank=Ranker(kind);before=parameter_hash(rank.model)
    warmup=warm(rank,tok,tau);started=time.monotonic();new=0
    run_id=str(time.time_ns());run={'utc':utc(),'model':kind,'resumed_records':len(prior),'tau':tau,'warmup':warmup,'code_sha256':sha(__file__)}
    try:
        with path.open('a' if resume else 'x',encoding='utf-8') as f:
            for task in tasks:
                ident=(task['id'],task['rep'],task['method'],task['batch_size'])
                if ident in keys:continue
                assert time.monotonic()-started<CFG['maximum_execution_seconds']
                result=request(rows[task['id']],rank,tok,task['method'],task['batch_size'],tau)
                result.update(rep=task['rep'],role=task['role'],run_id=run_id)
                f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');f.flush();keys.add(ident);new+=1
                if new%32==0:print(json.dumps({'phase':'main','model':kind,'completed':len(keys),'expected':len(tasks),'seconds':time.monotonic()-started}),flush=True)
        assert keys==expected and parameter_hash(rank.model)==before
        run.update(state='complete',records=len(keys),records_sha256=sha(path),parameters_unchanged=True,buffers_unchanged=buffer_hashes(rank.model)==rank.buffers,
                   parameter_hash=before,buffer_hashes=rank.buffers,seconds=time.monotonic()-started,
                   gpu=torch.cuda.get_device_name(0),peak_allocated_bytes=torch.cuda.max_memory_allocated())
        save(BASE/f'runtime_{kind}.json',run)
    except BaseException as exc:
        run.update(state='interrupted',error=repr(exc),completed=len(keys),seconds=time.monotonic()-started)
        save(BASE/f'interruption_{kind}_{run_id}.json',run);raise
    print(json.dumps({'state':'model_complete','model':kind,'records':len(keys)}),flush=True)


def reader(resume=False):
    torch=torch_init();gpu_check();verify_freeze();verify_assets()
    from transformers import AutoTokenizer,AutoModelForCausalLM
    folder=str(ROOT/'models/Qwen--Qwen3-8B');tok=AutoTokenizer.from_pretrained(folder,local_files_only=True)
    inputs={r['id']:r for r in read_jsonl(BASE/'inputs.jsonl')};prompts={};links=[]
    for kind in ['qwen','bge']:
        assert load(BASE/f'runtime_{kind}.json')['state']=='complete'
        for r in read_jsonl(BASE/f'requests_{kind}.jsonl'):
            question=inputs[r['id']]['question']
            messages=[{'role':'system','content':'You are a helpful question answering assistant.'},
                      {'role':'user','content':'Answer the question using the provided passages. Output only a concise answer, with no explanation. If the passages do not provide the answer, output Unanswerable.\n\nPassages:\n'+r['context']+'\n\nQuestion: '+question}]
            text,ids=qwen_prompt(tok,messages);assert len(ids)<=CFG['reader_max_input_tokens']
            pk=digest({'text':text,'max_new_tokens':CFG['reader_max_new_tokens']})
            prompts.setdefault(pk,{'key':pk,'text':text,'input_tokens':len(ids),'input_ids_sha256':digest(ids)})
            links.append({'model':kind,'id':r['id'],'method':r['method'],'batch_size':r['batch_size'],'rep':r['rep'],'key':pk})
    if not (BASE/'reader_links.jsonl').exists():save_rows(BASE/'reader_links.jsonl',links);save_rows(BASE/'reader_prompts.jsonl',prompts.values())
    else:assert read_jsonl(BASE/'reader_links.jsonl')==links
    model=AutoModelForCausalLM.from_pretrained(folder,torch_dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa',local_files_only=True).eval()
    path=BASE/'answers.jsonl';assert resume or not path.exists();old=read_jsonl(path) if path.exists() else []
    outputs={r['key']:r for r in old};assert len(outputs)==len(old) and set(outputs)<=set(prompts)
    options={'do_sample':False,'max_new_tokens':64,'pad_token_id':tok.eos_token_id,'repetition_penalty':1.0};started=time.monotonic()
    with path.open('a' if resume else 'x',encoding='utf-8') as f:
        for pk,prompt in prompts.items():
            if pk in outputs:continue
            gpu_check();features=tok(prompt['text'],add_special_tokens=False,return_tensors='pt').to('cuda')
            with torch.inference_mode():output=model.generate(**features,**options)
            ids=output[0,features.input_ids.shape[1]:].cpu().tolist()
            record={'key':pk,'output_ids':ids,'answer':tok.decode(ids,skip_special_tokens=True).strip(),'cap_hit':len(ids)==64}
            f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();outputs[pk]=record
            if len(outputs)%32==0:print(json.dumps({'phase':'reader','completed':len(outputs),'expected':len(prompts),'seconds':time.monotonic()-started}),flush=True)
    repeated=[]
    for pk,prompt in list(prompts.items())[:16]:
        gpu_check();features=tok(prompt['text'],add_special_tokens=False,return_tensors='pt').to('cuda')
        with torch.inference_mode():output=model.generate(**features,**options)
        ids=output[0,features.input_ids.shape[1]:].cpu().tolist()
        repeated.append({'key':pk,'output_ids':ids,'identical':ids==outputs[pk]['output_ids']})
    save_rows(BASE/'reader_repeats.jsonl',repeated)
    save(BASE/'reader_runtime.json',{'state':'complete','utc':utc(),'unique_prompts':len(prompts),'links':len(links),'seconds':time.monotonic()-started,
          'repeat_identity':sum(r['identical'] for r in repeated),'repeated_prompts':16,'cap_hits':sum(r['cap_hit'] for r in outputs.values()),
          'answers_sha256':sha(path),'memoized_exact_prompts':True})
    print(json.dumps({'state':'reader_complete','unique_prompts':len(prompts)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['calibration','execute','reader']);p.add_argument('--model',choices=['qwen','bge']);p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.phase=='execute':execute(args.model,args.resume)
    elif args.phase=='reader':reader(args.resume)
    else:calibration()
