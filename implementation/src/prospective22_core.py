import datetime
import hashlib
import itertools
import json
import math
import os

from common import ROOT, atomic_json, digest, normalize, read_jsonl
from cqc_budget12_core import file_sha, source_path, write_new
from precision21_core import decision


def config():
    cfg=json.loads((ROOT/'configs/prospective_stage22.json').read_text(encoding='utf-8'))
    return cfg,ROOT/cfg['output']


def select_ids(ids,cfg):
    pool=ids[320:512]
    assert len(ids)==512 and len(set(ids))==512 and len(pool)==192
    chosen=set(sorted(pool,key=lambda q:hashlib.sha256(('stage22|'+str(cfg['seed'])+'|'+q).encode()).hexdigest())[:cfg['questions']])
    return [q for q in pool if q in chosen]


def schedule(rows,cfg):
    permutations=list(itertools.permutations(cfg['arms']))
    result=[]
    for rep in range(cfg['repetitions']):
        ordered=sorted(enumerate(rows),key=lambda x:digest({'stage':22,'seed':cfg['seed'],'repeat':rep,'id':x[1]['id']}))
        for pos,(i,row) in enumerate(ordered):
            arms=permutations[i%len(permutations)]
            if rep%2:arms=arms[::-1]
            result.extend({'repeat':rep,'position':pos,'slot':slot,'id':row['id'],'arm':arm} for slot,arm in enumerate(arms))
    return result


def paired_exact(b,c):
    assert isinstance(b,int) and isinstance(c,int) and b>=0 and c>=0
    return sum(math.comb(b+c,k) for k in range(b,b+c+1))/(2**(b+c))


def wilson(k,n):
    assert 0<=k<=n and n>0
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]


def stability(pairs,engineering,cfg):
    cells={'both':0,'native_only':0,'fp32_only':0,'neither':0}
    for r in pairs:
        n=r['contrasts']['bf16_batch4_minus_bf16_serial']['context_changed']
        p=r['contrasts']['fp32_batch4_minus_fp32_serial']['context_changed']
        cells['both' if n and p else 'native_only' if n else 'fp32_only' if p else 'neither']+=1
    b,c=cells['native_only'],cells['fp32_only'];p=paired_exact(b,c)
    return {'contingency':cells,'one_sided_exact_mcnemar_p':p,'alpha':cfg['primary_alpha'],'primary_stability_confirmed':bool(engineering and b>c and p<=cfg['primary_alpha']),'unit':'question','first_repetition_only':True}


def check_prereg(assets=False):
    cfg,base=config();m=json.loads((base/'preregistration.json').read_text())
    for n,h in m['files_sha256'].items():assert file_sha(source_path(n))==h,n
    if assets:
        for n,h in m['external_asset_sha256'].items():assert file_sha(ROOT/n)==h,n
    return cfg,base,m


def preregister():
    cfg,base=config();assert not base.exists()
    old=json.loads((ROOT/'results/precision_stage21_used128/manifest.json').read_text())
    names=set(old['files_sha256'])
    names.update(['results/precision_stage21_used128/manifest.json','data/split_manifest.json','configs/prospective_stage22.json','research/stage22_human_review_record.md','research/stage22_prospective_protocol.md'])
    names.update('src/'+n for n in ['prospective22_core.py','prepare_prospective22.py','audit_retrieval22.py','run_prospective22.py','audit_prospective22.py','test_prospective22.py','queue_prospective22.py','check_prospective22_arithmetic.py','search_dense_corpus.py','search_embedding_corpus.py','retrieval_math.py'])
    external=['data/hotpot_development.jsonl','data/hotpot_pilot.jsonl','data/hotpot_training.jsonl','data/hotpot_training512.jsonl']
    external.extend(v+'/manifest.json' for v in cfg['indices'].values())
    for model in ['BAAI--bge-small-en-v1.5','intfloat--e5-base-v2']:
        external.extend(p.relative_to(ROOT).as_posix() for p in (ROOT/'models'/model).rglob('*') if p.is_file() and '.cache' not in p.parts)
    m={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files_sha256':{n:file_sha(source_path(n)) for n in sorted(names)},'external_asset_sha256':{n:file_sha(ROOT/n) for n in sorted(external)},'selected_queries_materialized':False,'new_retrieval_or_inference_started':False,'questions':cfg['questions'],'remaining_reserved_hotpot':128,'user_review_attestation':'research/stage22_human_review_record.md'}
    source=json.loads((ROOT/'data/split_manifest.json').read_text())['splits']['development']
    assert m['external_asset_sha256']['data/hotpot_development.jsonl']==source['sha256']
    base.mkdir();atomic_json(base/'preregistration.json',m)
    print(json.dumps({'state':'preregistered','sha256':file_sha(base/'preregistration.json')}),flush=True)


def split():
    cfg,base,m=check_prereg(assets=True)
    allrows=read_jsonl(ROOT/'data/hotpot_development.jsonl')
    ids=[r['id'] for r in allrows];chosen=select_ids(ids,cfg)
    assert ids==json.loads((ROOT/'data/split_manifest.json').read_text())['splits']['development']['ids']
    selected=[{'id':r['id'],'question':r['question']} for r in allrows if r['id'] in set(chosen)]
    assert [r['id'] for r in selected]==chosen and len(selected)==cfg['questions']
    previous=allrows[:320]
    for name in ['hotpot_pilot.jsonl','hotpot_training.jsonl','hotpot_training512.jsonl']:previous+=read_jsonl(ROOT/'data'/name)
    assert not set(chosen)&{r['id'] for r in previous}
    previous_hashes=sorted({digest(normalize(r['question'])) for r in previous})
    assert not {digest(normalize(r['question'])) for r in selected}&set(previous_hashes)
    folder=ROOT/cfg['data'];folder.mkdir(exist_ok=False)
    write_new(folder/'queries.jsonl',selected)
    atomic_json(folder/'split.json',{'preregistration_sha256':file_sha(base/'preregistration.json'),'source_sha256':file_sha(ROOT/'data/hotpot_development.jsonl'),'ids':chosen,'selection_rule':'hash64 from development[320:512], canonical development order','queries_sha256':file_sha(folder/'queries.jsonl'),'prior_normalized_question_hashes':previous_hashes,'selected_normalized_question_hashes':[digest(normalize(r['question'])) for r in selected],'id_overlap':0,'normalized_question_overlap':0,'labels_consulted_for_selection':False,'source_json_parser_reads_label_containing_rows':True,'gold_fields_exported':False,'remaining_reserved_ids':[q for q in ids[320:512] if q not in set(chosen)]})
    print(json.dumps({'state':'split','questions':len(selected),'remaining_reserved':192-len(selected)}),flush=True)


def freeze_inputs():
    cfg,base,m=check_prereg()
    assert not (base/'manifest.json').exists()
    rows=read_jsonl(base/'inputs.jsonl')
    assert len(rows)==cfg['questions'] and [r['id'] for r in rows]==[r['id'] for r in read_jsonl(ROOT/cfg['data']/'queries.jsonl')]
    write_new(base/'schedule.jsonl',schedule(rows,cfg))
    names=['inputs.jsonl','canonical_scores.jsonl','schedule.jsonl','retrieval_audit.json','preregistration.json']
    external=[cfg['data']+'/queries.jsonl',cfg['data']+'/split.json']
    for name in cfg['indices']:
        external.extend(cfg['output']+'/'+name+'/'+n for n in ['retrieved.jsonl','query_vectors.npy','runtime.json'])
    docs=sum(len(r['documents']) for r in rows);warm=sum(len(r['documents']) for r in rows[:cfg['warmup_questions']])
    out={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preregistration_sha256':file_sha(base/'preregistration.json'),'files_sha256':{**m['files_sha256'],**{n:file_sha(ROOT/n) for n in external}},'inputs_sha256':{n:file_sha(base/n) for n in names},'questions':len(rows),'documents':docs,'primary_generation_calls':len(rows)*len(cfg['arms'])*cfg['repetitions'],'primary_scored_documents':docs*len(cfg['arms'])*cfg['repetitions'],'warmup_generation_calls':cfg['warmup_questions']*len(cfg['arms']),'warmup_scored_documents':warm*len(cfg['arms']),'generation_started':False,'labels_exported':False}
    atomic_json(base/'manifest.json',out)
    print(json.dumps({'state':'input_frozen','sha256':file_sha(base/'manifest.json'),'documents':docs}),flush=True)


def check22():
    cfg,base,p=check_prereg();m=json.loads((base/'manifest.json').read_text())
    for n,h in m['files_sha256'].items():assert file_sha(source_path(n))==h,n
    for n,h in m['inputs_sha256'].items():assert file_sha(base/n)==h,n
    return cfg,base,m


def export_labels():
    cfg,base,m=check22()
    runtime=json.loads((base/'runtime.json').read_text());assert runtime['state']=='complete'
    p=json.loads((base/'preregistration.json').read_text())
    assert file_sha(ROOT/'data/hotpot_development.jsonl')==p['external_asset_sha256']['data/hotpot_development.jsonl']
    ids=[r['id'] for r in read_jsonl(base/'inputs.jsonl')];source={r['id']:r for r in read_jsonl(ROOT/'data/hotpot_development.jsonl')}
    write_new(ROOT/cfg['labels'],[source[q] for q in ids])
    atomic_json(ROOT/cfg['data']/'label_release.json',{'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'exported_after_inference','labels_sha256':file_sha(ROOT/cfg['labels']),'runtime_sha256':file_sha(base/'runtime.json'),'source_sha256':file_sha(ROOT/'data/hotpot_development.jsonl'),'questions':len(ids)})
