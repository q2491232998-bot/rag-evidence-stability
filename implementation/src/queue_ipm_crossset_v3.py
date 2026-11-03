import datetime
import json
import subprocess
import sys
import time

from filelock import FileLock

from common import ROOT, atomic_json
from ipm_crossset_v3_core import check_prereg
from cqc_budget12_core import file_sha


def main():
    cfg,base,_=check_prereg()
    py=sys.executable
    idle=[py,'src/run_idle_serial.py','--timeout','7200','--poll-seconds','15',py]
    commands=[('metric_check',[py,'src/verify_musique_metrics.py']),('split',[py,'src/prepare_ipm_crossset_v3.py','split'])]
    for name in ['bge','e5']:
        cmd=idle+['src/search_dense_corpus.py' if name=='bge' else 'src/search_embedding_corpus.py','--input',cfg['data']+'/queries.jsonl','--output',cfg['output']+'/'+name,'--top-k','200','--query-batch-size','64']
        if name=='e5':cmd+=['--index',cfg['indices'][name]]
        commands.append(('retrieve_'+name,cmd))
    commands.extend([('candidates',[py,'src/prepare_ipm_crossset_v3.py','candidates']),('retrieval_audit',[py,'src/audit_retrieval_ipm_v3.py']),('freeze_inputs',[py,'src/prepare_ipm_crossset_v3.py','freeze']),('preflight',[py,'src/audit_ipm_crossset_v3.py','preflight']),('benchmark',idle+['src/run_ipm_crossset_v3.py']),('labels',[py,'src/prepare_ipm_crossset_v3.py','labels']),('audit',[py,'src/audit_ipm_crossset_v3.py','full'])])
    with FileLock(str(ROOT/'logs/ipm_crossset_v3_queue.lock'),timeout=0):
        assert not list(base.glob('queue_*.json')),'No silent relaunch of an interrupted confirmation'
        run_id=str(time.time_ns());target=base/('queue_'+run_id+'.json')
        state={'state':'running','run_id':run_id,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preregistration_sha256':file_sha(base/'preregistration.json'),'steps':[]}
        for name,command in commands:
            if (ROOT/'logs/research.pause').exists():
                state.update(state='paused_by_existing_marker',stopped_before=name);atomic_json(target,state);return 75
            state['active_step']=name;atomic_json(target,state)
            log=ROOT/'logs'/('ipm_crossset_v3_'+run_id+'_'+name+'.log');started=time.monotonic()
            with log.open('x',encoding='utf-8') as stream:
                result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            state['steps'].append({'step':name,'exit_code':result.returncode,'seconds_including_idle_wait':time.monotonic()-started,'log':log.relative_to(ROOT).as_posix(),'log_sha256':file_sha(log)})
            print(json.dumps(state['steps'][-1]),flush=True)
            if result.returncode:
                state.update(state='failed',failed_step=name);atomic_json(target,state);return result.returncode
        state.update(state='complete',active_step=None,automatic_next_experiment=False,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        atomic_json(target,state);print(json.dumps({'state':'complete','automatic_next_experiment':False}),flush=True)
        return 0


if __name__=='__main__':sys.exit(main())
