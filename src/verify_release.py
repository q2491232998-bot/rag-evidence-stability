"""Replay the three v3 published value ledgers and the calibration audit (stdlib only)."""
import csv,gzip,collections,math,json,re,statistics,itertools
from pathlib import Path
R=Path(__file__).resolve().parents[1]; L=R/'paper/eswa_selective_rescoring_v1/generated'
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def eq(a,b):
 try:return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12)
 except (ValueError,TypeError):return str(a)==str(b)
def group(rows,fields):
 d=collections.defaultdict(list)
 for r in rows:d[tuple(r[k] for k in fields.split())].append(r)
 return d
def ok(v):return v=='True'
def col(rr,k):return [float(r[k]) for r in rr]
def main():
 cache={}; counts={}
 for r in read(L/'manuscript_evidence.csv'):
  p=r['source']
  if p not in cache:cache[p]=read(R/p)
  sel=json.loads(r['selector']) if r['selector'].startswith('{') else {'parameter':r['selector']};rr=[x for x in cache[p] if all(str(v)==x[k] for k,v in sel.items())];assert len(rr)==1 and eq(rr[0][r['field']],r['value']),r['key']
 counts['primary']=len(read(L/'manuscript_evidence.csv'))
 ext=group([r for r in read(R/'results/eswa_extension_20261111_v1/analysis/per_question_repeats.csv') if r['rep']=='0'],'model dataset method batch_size'); eps={'qwen':.9375,'bge':.1875}
 for r in read(L/'supplementary_derived_evidence.csv'):
  category,m,d,b,k=re.fullmatch(r'cl_(err|band)_(qwen|bge)_(hotpot|qasper)_(\d+)_(.*)',r['key']).groups()
  if category=='err':
   rr=ext[m,d,'band' if b=='4' else 'native',b];es=col(rr,'maximum_initial_error_to_bf16_serial'); val={'max':max(es),'exceed_q':sum(v>eps[m] for v in es),'equal_q':es.count(eps[m])}[k]
  else:
   rr=ext[m,d,'band',b];rat=col(rr,'time_ratio_to_serial');val={'slower_q':sum(v>1 for v in rat),'max_ratio':max(rat),'all_rescored_q':sum(x['rescored_count']==x['candidate_count'] for x in rr)}[k]
  assert eq(val,r['value']),(r['key'],val,r['value'])
 counts['supplementary']=len(read(L/'supplementary_derived_evidence.csv'))
 cal=R/'calibration4090'
 def ck(r):return (r['model'],r['dataset'],r['id'],r['batch_size'],r['rep'])
 cells={};nc=0
 with gzip.open(cal/'candidate_score_pairs.csv.gz','rt',encoding='utf-8',newline='') as f:
  for k,g in itertools.groupby(csv.DictReader(f),ck):
   assert k not in cells; gg=list(g);nc+=len(gg)
   for i,x in enumerate(gg):assert int(x['candidate_index'])==i and abs(float(x['batch_score'])-float(x['serial_score']))==float(x['absolute_error'])
   x=max(gg,key=lambda x:float(x['absolute_error']));cells[k]=(len(gg),x)
 for x in read(cal/'calibration_errors_by_batch_repeat.csv'):
  n,y=cells[ck(x)];assert int(x['candidate_count'])==n and all(eq(x[k],y[k]) for k in ['candidate_index','batch_score','serial_score']) and eq(x['max_abs_error'],y['absolute_error'])
 questions=group(read(cal/'calibration_errors_by_batch_repeat.csv'),'model dataset id'); modelmax={}
 for x in read(cal/'per_question_max.csv'):
  k=(x['model'],x['dataset'],x['id']);rr=questions[k];assert len(rr)==6 and max(col(rr,'max_abs_error'))==float(x['max_abs_error']);modelmax[k[0]]=max(float(x['max_abs_error']),modelmax.get(k[0],0))
 for x in read(cal/'pooled_epsilon_witnesses.csv'):assert modelmax[x['model']]*1.5==float(x['recalibrated_epsilon'])
 assert len(cells)==768 and len(questions)==128;counts['calibration_pairs']=nc
 new=R/'results/rtx4090_validation'; raw=[x for x in read(new/'per_question.csv') if x['rep']=='0'];ev=group(raw,'model dataset method batch_size'); cross=group(read(new/'cross_hardware_per_question.csv'),'model dataset'); values={}
 for (m,d,method,b),rr in ev.items():
  sid={r['id']:r for r in ev[m,d,'serial','1']};p=f'vf_{m}_{d}_'
  if method=='serial':
   values[p+'n']=len(rr);values[p+'serial_f1']=statistics.mean(col(rr,'answer_f1'));xx=cross[m,d];values[p+'xhw_n']=len(xx);values[p+'xhw']=sum(not ok(x['context_equal_3090ti_rep0']) for x in xx);values[p+'xhw_maxdiff']=max(col(xx,'max_score_difference_3090ti_rep0'));continue
  common={'cd':sum(x['context_sha256']!=sid[x['id']]['context_sha256'] for x in rr),'ans':sum(x['answer_sha256']!=sid[x['id']]['answer_sha256'] for x in rr),'resc':statistics.mean(col(rr,'rescored_fraction')),'time':sum(col(rr,'two_repeat_median_seconds'))/sum(col(rr,'serial_two_repeat_median_seconds')),'slower':sum(ok(x['slower_than_serial']) for x in rr),'maxratio':max(col(rr,'paired_time_ratio_to_serial')),'f1':statistics.mean(col(rr,'answer_f1')),'maxerr':max(col(rr,'max_abs_batch_serial_error'))}
  if method.startswith('band'):
   common.update(exq=sum(ok(x['question_exceeds_epsilon']) for x in rr),exc=sum(col(rr,'candidates_exceeding_epsilon')));values[f'vf_{m}_{method}_eps']=float(rr[0]['epsilon'])
  for k,v in common.items():values[f'{p}{method}_{b}_{k}']=v
 for m in ('qwen','bge'):
  values[f'vf_{m}_recal_calmax']=modelmax[m];values[f'vf_{m}_eval_maxerr']=max(col([x for x in raw if x['model']==m],'max_abs_batch_serial_error'))
 for n in (192,200):values[f'vf_ub_{n}']=(1-.05**(1/n))*100
 for s in ('dev','test'):values[f'vf_qasper_{s}_n']=sum(x['source_split']==s for x in ev['qwen','qasper','serial','1'])
 for r in read(L/'rtx4090_validation_evidence.csv'):
  k=r['key']
  if k.endswith('_range'):
   prefix,s=k[:-6].rsplit('_',1);vv=[int(values[f'{prefix}_{b}_{s}']) for b in (4,8,32)];a,b=min(vv),max(vv);assert (str(a) if a==b else f'{a}--{b}')==r['display'],k
  else:assert k in values and eq(values[k],r['value']),(k,values.get(k),r['value'])
 counts['rtx4090']=len(read(L/'rtx4090_validation_evidence.csv'));print('Verified',counts)
if __name__=='__main__':main()

