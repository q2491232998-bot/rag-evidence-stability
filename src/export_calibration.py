"""Export score pairs for every native calibration candidate; verify frozen witnesses. Python stdlib."""
import argparse,csv,gzip,hashlib,json,itertools,collections
from pathlib import Path
def dump(path,rows):
 with path.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=True)
 freeze=json.loads((a.input/'epsilon_freeze.json').read_text()); targets=json.loads((a.input/'epsilon_recalibrated.json').read_text())['models']
 expected=[json.loads(x) for x in (a.input/'calibration_errors.jsonl').read_text().splitlines()]
 def key(r):return r['model'],r['id'],r['batch_size'],r['rep']
 index={key(r):r for r in expected};assert len(index)==len(expected)==768
 witnesses=[];rows=[];n=0;ids={}
 with gzip.open(a.output/'candidate_score_pairs.csv.gz','wt',encoding='utf-8',newline='') as f:
  w=csv.writer(f);w.writerow('model dataset id batch_size rep candidate_index batch_score serial_score absolute_error'.split())
  for m in ('qwen','bge'):
   p=a.input/('calibration_'+m+'.jsonl');h=hashlib.sha256(p.read_bytes()).hexdigest(); assert [v for k,v in freeze['calibration_records_sha256'].items() if k.endswith('/'+p.name)]==[h]
   allr=[json.loads(line) for line in p.read_text().splitlines()];rr={(r['id'],r['method'],r['batch_size'],r['rep']):r for r in allr}; assert len(allr)==len(rr)==512
   ids[m]=set((r['dataset'],r['id']) for r in allr); assert collections.Counter(d for d,i in ids[m])=={'qasper':32,'hotpot':32}
   for d,i in sorted(ids[m]):
    cells=[];observations=0
    for b,rep in itertools.product((4,8,32),(0,1)):
     ref=rr[i,'serial',1,rep]; r=rr[i,'native',b,rep]; assert r['model']==ref['model']==m and r['encoded_sha256']==ref['encoded_sha256']
     sc=r['scores'];se=ref['scores'];assert len(sc)==len(se)==r['candidate_count']==ref['candidate_count']
     diff=[abs(x-y) for x,y in zip(sc,se)]; ix=diff.index(max(diff)); v=index[m,i,b,rep];assert (v['dataset'],v['max_abs_error'],v['candidate_index'],v['batch_score'],v['serial_score'])==(d,diff[ix],ix,sc[ix],se[ix])
     for j,(x,y,e) in enumerate(zip(sc,se,diff)):w.writerow((m,d,i,b,rep,j,x,y,e))
     observations+=len(diff); cells.append(v)
    top=max(cells,key=lambda x:x['max_abs_error']);rows.append(dict(top,score_pair_observations=observations,tied_configurations=sum(c['max_abs_error']==top['max_abs_error'] for c in cells),configurations=6));n+=observations
   mm=max(r['max_abs_error'] for r in rows if r['model']==m); spec=targets[m]; assert mm==spec['maximum_absolute_error'] and mm*1.5==spec['epsilon']
   orig=spec['witness']; assert index[key(orig)]==orig; witnesses.append(dict(orig,multiplier=1.5,recalibrated_epsilon=mm*1.5,source_jsonl_sha256=h))
 assert ids['qwen']==ids['bge'];dump(a.output/'per_question_max.csv',rows);dump(a.output/'pooled_epsilon_witnesses.csv',witnesses)
 print(f'passed: {n} aligned score pairs; 768 cells; 128 question/model rows; 64 shared questions.')
if __name__=='__main__':main()

