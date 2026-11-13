"""Explicit timing-only recovery of all requests in a resource-interrupted invocation."""
import time
from eswa_stage2_common import *
from run_eswa_stage2_v1 import torch_init,gpu_check,verify_freeze,verify_assets,Ranker,parameter_hash,buffer_hashes,warm,request


def key_(r):return [r['model'],r['id'],r['method'],r['batch_size'],r['rep']]


def main():
    torch=torch_init();gpu_check();verify_freeze();verify_assets()
    from transformers import AutoTokenizer
    plan=load(BASE/'timing_recovery_plan.json')
    for path,want in plan['code_sha256'].items():assert sha(ROOT/path)==want,path
    original={tuple(key_(r)):r for r in read_jsonl(BASE/'requests_qwen.jsonl')};rows={r['id']:r for r in read_jsonl(BASE/'inputs.jsonl')}
    for entry in plan['requests']:assert digest(original[tuple(entry['key'])])==entry['original_record_digest']
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen--Qwen3-8B'),local_files_only=True)
    rank=Ranker('qwen');before=parameter_hash(rank.model);warmup=warm(rank,tok,CFG['qwen_tau']);start=time.monotonic()
    path=BASE/'timing_recovery_qwen.jsonl';run={'utc':utc(),'state':'running','warmup':warmup,'plan_sha256':sha(BASE/'timing_recovery_plan.json')}
    completed=0;differences=0
    try:
        with path.open('x',encoding='utf-8') as f:
            for entry in plan['requests']:
                previous=original[tuple(entry['key'])]
                result=request(rows[previous['id']],rank,tok,previous['method'],previous['batch_size'],CFG['qwen_tau'])
                gpu_check()
                fields=['encoded_sha256','initial_scores','scores','order','selected','context','rescored_indices','rescored_count']
                changed=[name for name in fields if result[name]!=previous[name]];differences+=bool(changed)
                result.update(rep=previous['rep'],role='timing_recovery',original_run_id=previous['run_id'],
                              original_total_seconds=previous['total_seconds'],changed_fields=changed,post_request_gpu_check=True)
                f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');f.flush();completed+=1
                if completed%32==0:print(json.dumps({'phase':'timing_recovery','model':'qwen','completed':completed,'expected':len(plan['requests']),'seconds':time.monotonic()-start}),flush=True)
        assert parameter_hash(rank.model)==before and buffer_hashes(rank.model)==rank.buffers
        run.update(state='complete',records=completed,changed_output_records=differences,parameters_unchanged=True,buffers_unchanged=True,
                   parameter_hash=before,records_sha256=sha(path),seconds=time.monotonic()-start)
        save(BASE/'timing_recovery_runtime.json',run)
    except BaseException as exc:
        run.update(state='interrupted',error=repr(exc),completed=completed,seconds=time.monotonic()-start)
        save(BASE/'timing_recovery_interruption.json',run);raise
    print(json.dumps({'state':'timing_recovery_complete','records':completed,'changed_output_records':differences}),flush=True)


if __name__=='__main__':main()
