"""Prospectively frozen MuSiQue question transfer in the existing fixed corpus."""
import ast,datetime,hashlib,itertools,json,os
if os.name=='nt':raise SystemExit('Remote experiment operations only.')
from common import ROOT,atomic_json,normalize,read_jsonl
from cqc_budget12_core import file_sha,write_new
from prospective22_core import stability,wilson,decision

def config():
    cfg=json.loads((ROOT/'configs/ipm_crossset_v3.json').read_text())
    return cfg,ROOT/cfg['output']

def schedule(rows,cfg):
    permutations=list(itertools.permutations(cfg['arms']));out=[]
    for rep in range(cfg['repetitions']):
        ordered=sorted(enumerate(rows),key=lambda x:hashlib.sha256(f"ipm-crossset-v3|{cfg['seed']}|{rep}|{x[1]['id']}".encode()).hexdigest())
        for pos,(i,row) in enumerate(ordered):
            arms=permutations[i%24]
            if rep%2:arms=arms[::-1]
            out.extend({'repeat':rep,'position':pos,'slot':slot,'id':row['id'],'arm':arm} for slot,arm in enumerate(arms))
    return out

def preregister():
    cfg,base=config();assert not base.exists()
    audit=json.loads((ROOT/'results/ipm_crossset_v3_audit/summary.json').read_text())
    assert audit['state']=='passed' and audit['eligible_count']>=cfg['questions']
    names={'configs/ipm_crossset_v3.json','configs/qwen512_v1.json','research/ipm_crossset_protocol_v3.md',
           'results/ipm_crossset_v3_audit/summary.json','results/ipm_crossset_v3_audit/file_inventory.json',
           'results/ipm_crossset_v3_audit/used_question_hashes.json',
           'data/musique_v1/manifest.json','data/musique_v1/reserved256_questions.jsonl'}
    pending=['ipm_crossset_v3_core','prepare_ipm_crossset_v3','audit_retrieval_ipm_v3',
             'run_ipm_crossset_v3','audit_ipm_crossset_v3','queue_ipm_crossset_v3',
             'search_dense_corpus','search_embedding_corpus','musique_metrics','verify_musique_metrics','run_idle_serial']
    seen=set()
    while pending:
        module=pending.pop();p=ROOT/'src'/f'{module}.py'
        if module in seen or not p.exists():continue
        seen.add(module);names.add(p.relative_to(ROOT).as_posix())
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node,ast.ImportFrom) and node.module:pending.append(node.module.split('.')[0])
            elif isinstance(node,ast.Import):pending.extend(n.name.split('.')[0] for n in node.names)
    names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'references/musique_official').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    external=[v+'/manifest.json' for v in cfg['indices'].values()]
    for model in ['BAAI--bge-small-en-v1.5','intfloat--e5-base-v2']:
        external.extend(p.relative_to(ROOT).as_posix() for p in (ROOT/'models'/model).rglob('*') if p.is_file() and '.cache' not in p.parts)
    expected=json.loads((ROOT/'data/musique_v1/manifest.json').read_text())['splits']['reserved256']
    assert file_sha(ROOT/'data/musique_v1/reserved256_questions.jsonl')==expected['blind_questions_sha256']
    assert file_sha(ROOT/'data/musique_v1/reserved256.jsonl')==expected['labeled_sha256']
    m={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'registration_scope':'Internal timestamped protocol and source freeze, not public preregistration.',
       'files_sha256':{n:file_sha(ROOT/n) for n in sorted(names)},
       'external_asset_sha256':{n:file_sha(ROOT/n) for n in sorted(external)},
       'label_source_sha256':expected['labeled_sha256'],'selected_queries_materialized':False,
       'new_retrieval_or_inference_started':False,'questions':cfg['questions'],
       'source_restoration':'Previously retained local prepared files transferred to remote; both historical hashes verified. Original labels were prepared in October; not used for this selection.'}
    base.mkdir();atomic_json(base/'preregistration.json',m)
    print(json.dumps({'state':'protocol_frozen','sha256':file_sha(base/'preregistration.json'),'source_files':len(names)}),flush=True)

def check_prereg(assets=False):
    cfg,base=config();m=json.loads((base/'preregistration.json').read_text())
    for n,h in m['files_sha256'].items():assert file_sha(ROOT/n)==h,n
    if assets:
        for n,h in m['external_asset_sha256'].items():assert file_sha(ROOT/n)==h,n
    return cfg,base,m

def split():
    cfg,base,m=check_prereg(assets=True)
    audit=json.loads((ROOT/'results/ipm_crossset_v3_audit/summary.json').read_text())
    rows=read_jsonl(ROOT/'data/musique_v1/reserved256_questions.jsonl');eligible=set(audit['eligible_ids'])
    chosen=set(sorted(eligible,key=lambda q:hashlib.sha256(f"ipm-crossset-v3|{cfg['seed']}|{q}".encode()).hexdigest())[:cfg['questions']])
    rows=[r for r in rows if r['id'] in chosen];assert len(rows)==cfg['questions']
    prior=set(json.loads((ROOT/'results/ipm_crossset_v3_audit/used_question_hashes.json').read_text()))
    selected=[hashlib.sha256(normalize(r['question']).encode()).hexdigest() for r in rows]
    assert not prior&set(selected)
    folder=ROOT/cfg['data'];folder.mkdir(exist_ok=False);write_new(folder/'queries.jsonl',rows)
    atomic_json(folder/'split.json',{'ids':[r['id'] for r in rows],
       'selection':'SHA256 ipm-crossset-v3|20261102|ID, first 96 eligible; retain original reserved partition order.',
       'queries_sha256':file_sha(folder/'queries.jsonl'),'selected_normalized_question_hashes':selected,
       'prior_normalized_question_hashes':sorted(prior),'labels_consulted':False,
       'question_id_overlap':0,'normalized_question_overlap':0,'component_disjointness_claimed':False})
    print(json.dumps({'state':'split','questions':len(rows),'labels_read':False}),flush=True)

def freeze_inputs():
    cfg,base,m=check_prereg();assert not (base/'manifest.json').exists()
    rows=read_jsonl(base/'inputs.jsonl');assert len(rows)==cfg['questions']
    assert [{'id':r['id'],'question':r['question']} for r in rows]==read_jsonl(ROOT/cfg['data']/'queries.jsonl')
    write_new(base/'schedule.jsonl',schedule(rows,cfg))
    names=['inputs.jsonl','canonical_scores.jsonl','schedule.jsonl','retrieval_audit.json','preregistration.json']
    ext=[cfg['data']+'/queries.jsonl',cfg['data']+'/split.json']
    for name in cfg['indices']:ext.extend(cfg['output']+'/'+name+'/'+n for n in ['retrieved.jsonl','query_vectors.npy','runtime.json'])
    docs=sum(len(r['documents']) for r in rows);warm=sum(len(r['documents']) for r in rows[:2])
    atomic_json(base/'manifest.json',{'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'files_sha256':{**m['files_sha256'],**{n:file_sha(ROOT/n) for n in ext}},
      'inputs_sha256':{n:file_sha(base/n) for n in names},'questions':len(rows),'documents':docs,
      'primary_generation_calls':len(rows)*8,'primary_scored_documents':docs*8,
      'warmup_generation_calls':8,'warmup_scored_documents':warm*4,'generation_started':False,'labels_exported':False})
    print(json.dumps({'state':'inputs_frozen','documents':docs}),flush=True)

def check22():
    cfg,base,_=check_prereg();m=json.loads((base/'manifest.json').read_text())
    for n,h in m['files_sha256'].items():assert file_sha(ROOT/n)==h,n
    for n,h in m['inputs_sha256'].items():assert file_sha(base/n)==h,n
    return cfg,base,m

def export_labels():
    cfg,base,m=check22();runtime=json.loads((base/'runtime.json').read_text())
    assert runtime['state']=='complete';p=json.loads((base/'preregistration.json').read_text())
    source=ROOT/'data/musique_v1/reserved256.jsonl';assert file_sha(source)==p['label_source_sha256']
    rows={r['id']:r for r in read_jsonl(source)};ids=[r['id'] for r in read_jsonl(base/'inputs.jsonl')]
    write_new(ROOT/cfg['labels'],[{k:rows[q][k] for k in ['id','question','answer','answer_aliases','type']} for q in ids])
    atomic_json(ROOT/cfg['data']/'label_release.json',{'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'state':'exported_after_inference','labels_sha256':file_sha(ROOT/cfg['labels']),
      'runtime_sha256':file_sha(base/'runtime.json'),'source_sha256':file_sha(source),'questions':len(ids)})
