"""Remote-only editorial export from already audited CSVs; no model execution."""
import csv, json, os, re, hashlib, shutil
from pathlib import Path
assert os.name != 'nt', 'Remote-only evidence analysis/build'
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'paper/eswa_selective_rescoring_v1'; P.mkdir(parents=True,exist_ok=True)
G=P/'generated'; G.mkdir(exist_ok=True)
E=ROOT/'results/eswa_extension_20261111_v1/analysis'
S=ROOT/'results/eswa_gate_stage2_v1/analysis'
D=ROOT/'results/eswa_gate_20261110_v1/stage1'
ledger=[]; macros={}
def read(p): return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def writecsv(p,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def record(key,value,path,selector,field,transform='identity',display=None):
    display=str(value) if display is None else display
    assert key not in macros,key
    macros[key]=display
    ledger.append(dict(key=key,value=value,display=display,source=str(path.relative_to(ROOT)) if isinstance(path,Path) else path,selector=selector,field=field,transformation=transform))
    return display
def cell(key,row,field,path,fmt=None):
    v=row[field]; disp=fmt(float(v)) if fmt and v!='' else v
    return record(key,v,path,json.dumps({k:row[k] for k in ['model','dataset','method','batch_size','rep','split','tau','round_step','seed','source'] if k in row}),field,'formatted only' if fmt else 'identity',disp)
def pct(v):return f'{100*v:.2f}'
def ratio(v):return f'{v:.3f}'
names={'qwen':'Qwen','bge':'BGE','qasper':'QASPER','hotpot':'HotpotQA','serial':'Serial','batch4':'Native','native':'Native','gate_v1':'Gate v1','gate_v2':'Gate v2','top15':'Top-15','band':'Band','fp32_math_batch4':'FP32/MATH','fp32_math_serial':'FP32 serial'}
def table(file,caption,label,headers,rows,align=None):
    align=align or ('l'*len(headers))
    txt='\\begin{table}[!htbp]\n\\centering\\small\n\\caption{'+caption+'}\\label{'+label+'}\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{'+align+'}\n\\toprule\n'+' & '.join(headers)+r' \\'+ '\n\\midrule\n'
    txt+='\n'.join(' & '.join(map(str,r))+r' \\' for r in rows)
    txt+='\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    (G/file).write_text(txt)
main=read(S/'main_results.csv'); ext=read(E/'main_results.csv')
for prefix,rows,path in [('base',main,S/'main_results.csv'),('ext',ext,E/'main_results.csv')]:
    for r in rows:
        key='_'.join([prefix,r['model'],r['dataset'],r['method'],r['batch_size']])
        for field in ['n','context_differences','member_differences','mean_rescored_fraction','max_rescored_fraction','gold_recall','answer_f1','aggregate_time_ratio_to_serial','zero_change_one_sided95_upper']:
            if r.get(field,'')!='':cell(key+'_'+field,r,field,path,pct if field in ['mean_rescored_fraction','max_rescored_fraction','gold_recall','answer_f1','zero_change_one_sided95_upper'] else ratio if field=='aggregate_time_ratio_to_serial' else None)

rows=[]
for kind in ['qwen','bge']:
    for ds in ['qasper','hotpot']:
        rr=[r for r in main if r['model']==kind and r['dataset']==ds and r['method'] in ['serial','batch4','gate_v1','gate_v2','top15']]
        rr.sort(key=lambda r:['serial','batch4','gate_v1','gate_v2','top15'].index(r['method']))
        for j,r in enumerate(rr):
            rows.append([names[kind]+'/'+names[ds] if j==0 else '',names[r['method']],r['context_differences'],r['member_differences'],pct(float(r['mean_rescored_fraction'])),ratio(float(r['aggregate_time_ratio_to_serial'])),pct(float(r['gold_recall'])),pct(float(r['answer_f1']))])
table('batch4.tex','Fixed batch-4 evaluation after the gate rules were frozen. Context and member differences are relative to BF16 serial; R is the mean percentage of candidates rescored. Time is the ratio of sums of per-question two-run medians to serial. Recall and F1 are percentages. QASPER has 192 questions (172 with nonempty textual evidence); HotpotQA has 200. These same questions are reused in the extension.','tab:batch4',['Model/data','Method','$D_C$','$D_M$','R (\\%)','Time','Recall','F1'],rows,'llrrrrrr')
rows=[]
for kind in ['qwen','bge']:
    for ds in ['qasper','hotpot']:
        for b in [4,8,32]:
            r=next(r for r in ext if r['model']==kind and r['dataset']==ds and r['method']=='band' and int(r['batch_size'])==b)
            rows.append([names[kind]+'/'+names[ds],b,r['context_differences'],pct(float(r['mean_rescored_fraction'])),pct(float(r['max_rescored_fraction'])),ratio(float(r['aggregate_time_ratio_to_serial'])),r['epsilon_exceeded_candidates']])
table('band.tex','Band recovers the serial context in every tested cell. R reports mean and single-question maximum rescoring percentages; V counts candidates exceeding the fixed epsilon in the first repeat (also zero in the second). Repeats do not increase the independent sample size. Quality equals the corresponding serial row in Table \\ref{tab:batch4}.','tab:band',['Model/data','Batch','$D_C$','R mean','R max','Time','V'],rows,'llrrrrr')
for kind in ['qwen','bge']:
    for ds in ['qasper','hotpot']:
        rows=[]
        for b in [2,8,16,32]:
            for method in ['native','gate_v2','top15']:
                r=next(r for r in ext if r['model']==kind and r['dataset']==ds and r['method']==method and int(r['batch_size'])==b)
                rows.append([b,names[method],r['context_differences'],r['member_differences'],pct(float(r['mean_rescored_fraction'])),pct(float(r['max_rescored_fraction'])),ratio(float(r['aggregate_time_ratio_to_serial'])),pct(float(r['gold_recall'])),pct(float(r['answer_f1']))])
        table(f'scan_{kind}_{ds}.tex',f'All additional batch sizes for {names[kind]} on {names[ds]}. R mean/max are percentages. Quality is from the first repeat; both repeats have identical scores and contexts. Time uses both effective repeats; full per-repeat outcomes and invalid attempts remain in the CSV supplement.',f'tab:scan{kind}{ds}',['Batch','Method','$D_C$','$D_M$','R mean','R max','Time','Recall','F1'],rows,'llrrrrrrr')

mot=read(E/'batch_motivation_summary.csv');print('motivation schema',mot[0])
prec=read(E/'precision_summary.csv');rand=read(E/'random_grouping_summary.csv')
for typ,rr,path in [('mot',mot,E/'batch_motivation_summary.csv'),('prec',prec,E/'precision_summary.csv'),('random',rand,E/'random_grouping_summary.csv')]:
    for i,r in enumerate(rr):
        for f,v in r.items():
            if re.fullmatch(r'-?\d+(\.\d+)?([eE][-+]?\d+)?',v or ''):record(f'{typ}_{i}_{f}',v,path,json.dumps(r),f)
pr=[]
for r in prec:
    if r['batch_rep']!='0':continue
    pr.append([names[r['model']]+'/'+names[r['dataset']],r['context_differences_fp32_batch4_vs_fp32_serial'],r['context_differences_fp32_serial_vs_bf16_serial'],ratio(float(r['fp32_serial_time_ratio_to_bf16_serial'])),ratio(float(r['fp32_batch4_time_ratio_to_bf16_serial'])),r['score_candidates_different'],f"{float(r['maximum_score_difference']):.2e}"])
table('precision.tex','FP32/MATH changes the reference. The first difference column compares FP32 batch 4 with FP32 serial; the second compares FP32 serial with BF16 serial. TS/TB are FP32 serial/batch-4 time ratios to BF16 serial. FP32 serial is one run; batch 4 and the denominator use two-run medians from the earlier session. Score counts and maximum errors compare FP32 batch 4 with FP32 serial; both batch repeats agree on these results.','tab:precision',['Model/data','$D_{32,32}$','$D_{32,16}$','TS','TB','Score diff.','Max. error'],pr,'lrrrrrr')
rr=[]
for ds in ['qasper','hotpot']:
    for seed in ['17','29','43']:
        a=next(r for r in rand if r['dataset']==ds and r['seed']==seed and r['method']=='random_native')
        b=next(r for r in rand if r['dataset']==ds and r['seed']==seed and r['method']=='random_gate_v2')
        rr.append([names[ds],seed,a['n'],a['context_differences'],b['context_differences']])
table('random.tex','Random candidate grouping at Qwen batch 8. Every seed is reported. Counts are context differences from BF16 serial; these runs assess context identity only.','tab:random',['Data','Seed','Questions','Native','Gate v2'],rr,'lrrrr')
mr=[]
for r in mot:
    mr.append([names[r['model']]+'/'+names[r['dataset']], 'Original' if r['source'].startswith('stage2') else 'Extension',r['batch_sizes'].replace('[','').replace(']','').replace(' ',''),r['n'],r['context_varies_questions'],r['answer_varies_questions'],r['normalized_answer_varies_questions']])
table('motivation.tex','Batch size changes the context and sometimes the answer for the same question. Each count is the number of questions with more than one distinct value across the listed native batch sizes, recomputed from per-question CSVs. Raw and normalized answer strings are distinguished. Original and extension sessions are not pooled.','tab:motivation',['Model/data','Session','Batches','n','Context','Answer','Norm.'],mr,'lllrrrr')

# Copy summary CSVs as a compact audit interface, retaining source hashes.
source_manifest=[]
for folder,source in [('fixed_batch4',S),('extension',E),('development',D)]:
    dest=G/folder;dest.mkdir(exist_ok=True)
    for p in source.glob('*.csv'):
        if p.stat().st_size<150000:
            shutil.copyfile(p,dest/p.name)
            source_manifest.append(dict(file=f'{folder}/{p.name}',source=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))

dev=read(D/'selected_results.csv')
dr=[]
for r in dev:
    if r['model']=='qwen' and r['dataset']=='qasper' and r['split']=='test':
        method=r['method'].replace('_',' ')
        dr.append([method,r['tau'] or r['round_step'] or '--',r['context_differences'],r['member_differences'],pct(float(r['mean_rescored_fraction'])),pct(float(r['gold_recall_mean']))])
table('development.tex','Complete selected-control comparison on the 128-question Qwen development diagnostic set. R and evidence recall are percentages. These are CPU simulations with saved scores, not measured GPU latency. For rounding, differences here use the unrounded BF16 serial reference; the separate rounded-reference comparisons and every threshold for both development rerankers are retained in the CSV supplement.','tab:development',['Method','Parameter','$D_C$','$D_M$','R','Recall'],dr,'llrrrr')
for method in ['batch4','gated','top15']:
    matches=[r for r in dev if r['model']=='qwen' and r['dataset']=='qasper' and r['split']=='test' and r['method']==method]
    if not matches:continue
    r=matches[0]
    for f in ['n','context_differences','context_difference_reduction','mean_rescored_fraction']:
        cell('dev_'+method+'_'+f,r,f,D/'selected_results.csv',pct if 'fraction' in f or 'reduction' in f else None)

cfg=json.loads((ROOT/'configs/eswa_gate_stage2_v1.json').read_text())
params=[]
for k in ['qasper_evaluation_papers','bge_calibration_papers','hotpot_dev_questions','qasper_candidate_cap','reranker_max_length','context_tokens','top_k','gate_window','qwen_tau','reader_max_new_tokens','reader_max_input_tokens','repetitions','seed']:
    params.append(dict(parameter=k,value=cfg[k],source='configs/eswa_gate_stage2_v1.json',meaning='frozen protocol parameter'))
for k,v in [('tau_grid',json.dumps(cfg['tau_grid'])),('native_scan_batches',json.dumps(cfg['scan_batches'])),('extension_native_batches','2,8,16,32'),('band_batches','4,8,32'),('random_seeds','17,29,43'),('rounding_steps','0.125,0.25,0.5'),('pytorch_version','2.5.1+cu121'),('transformers_version','4.57.6'),('qasper_nonempty_evidence_n',next(r['gold_recall_n'] for r in main if r['dataset']=='qasper' and r['method']=='serial')),('gpu_name','NVIDIA GeForce RTX 3090 Ti')]:
    params.append(dict(parameter=k,value=v,source='frozen configs / source / runtime / main_results.csv',meaning='configuration or original result denominator'))
for k,v in [('bge_tau',.125),('top_rescore_k',15),('epsilon_multiplier',1.5),('original_development_calibration',64),('original_development_diagnostic',128),('old_hotpot_observations',64),('old_musique_observations',96),('target_reduction_percent',90),('target_cost_percent',20),('confidence_percent',95),('codex_desktop_version','26.1103.7773.0')]:
    params.append(dict(parameter=k,value=v,source='frozen protocols / historical sampling / installed package receipt',meaning='design or provenance, not a new empirical result'))
audit=json.loads((E/'audit.json').read_text());sample=json.loads((ROOT/'results/eswa_gate_stage2_v1/sampling.json').read_text());candidates=json.loads((ROOT/'results/eswa_gate_stage2_v1/candidate_manifest.json').read_text())
for k,v in audit['counts'].items():params.append(dict(parameter=k,value=v,source='extension audit.json',meaning='independent saved-trace validation'))
for k,v in audit['reader_runtime'].items():
    if isinstance(v,(float,int)):params.append(dict(parameter='reader_'+k,value=v,source='extension audit.json',meaning='exact prompt cache protocol; seconds includes waiting'))
for k,v in audit['gpu_occupancy'].items():
    if isinstance(v,(float,int)):params.append(dict(parameter='gpu_'+k,value=v,source='extension audit.json',meaning='sampled GPU occupancy'))
for k in ['qasper_paper_overlap','qasper_id_overlap','qasper_generic_question_text_overlap','hotpot_id_text_overlap']:
    params.append(dict(parameter=k,value=sample[k],source='sampling.json',meaning='history overlap audit'))
params.append(dict(parameter='retrieval_corpus_documents',value=candidates['retrieval_corpus_documents'],source='candidate_manifest.json',meaning='Hotpot retrieval corpus'))
writecsv(G/'protocol_and_audit.csv',params)
for r in params:record('param_'+r['parameter'],r['value'],G/'protocol_and_audit.csv',r['parameter'],'value')
for i,r in enumerate(read(E/'epsilon_summary.csv')):
    for f in ['epsilon','development_maximum_error','questions','candidates']:
        cell('epsilon_'+r['model']+'_'+f,r,f,E/'epsilon_summary.csv')

# Figure: actual cost versus residual evidence changes, all specified settings.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(10.5,6.8),sharex=True)
colors={'native':'#777777','gate_v2':'#0072B2','top15':'#D55E00','band':'#009E73'}
markers={'native':'x','gate_v2':'o','top15':'s','band':'^'}
for ax,(kind,ds) in zip(axes.flat,[(m,d) for m in ['qwen','bge'] for d in ['qasper','hotpot']]):
    for method in ['native','gate_v2','top15','band']:
        rr=sorted([r for r in ext if r['model']==kind and r['dataset']==ds and r['method']==method],key=lambda r:int(r['batch_size']))
        x=[float(r['aggregate_time_ratio_to_serial']) for r in rr];y=[int(r['context_differences']) for r in rr]
        ax.scatter(x,y,color=colors[method],marker=markers[method],label=names[method],s=46,alpha=.85)
    ax.set_title(names[kind]+' / '+names[ds]);ax.axvline(1,color='#bbbbbb',ls='--',lw=.8);ax.set_ylim(bottom=-2);ax.grid(alpha=.18);ax.set_ylabel('Changed contexts (questions)')
for ax in axes[1]:ax.set_xlabel('Reranker time / BF16 serial time')
axes[0,0].legend(frameon=False,fontsize=9,ncol=2)
fig.tight_layout();fig.savefig(G/'cost_preservation.pdf',bbox_inches='tight');fig.savefig(G/'cost_preservation.png',dpi=160,bbox_inches='tight');plt.close(fig)
writecsv(G/'manuscript_evidence.csv',ledger)
(G/'values.tex').write_text('\n'.join('\\expandafter\\def\\csname S'+k+'\\endcsname{'+v+'}' for k,v in macros.items()))
(G/'source_manifest.json').write_text(json.dumps(source_manifest,indent=2))
print(json.dumps({'state':'evidence_exported','numeric_ledger_rows':len(ledger),'source_summaries':len(source_manifest)}))
