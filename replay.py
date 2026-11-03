"""Replay the shared numerical projections without models or original text."""
from collections import Counter,defaultdict
import hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return [json.loads(s) for s in p.read_text().splitlines()]
out={}
for panel in ['S22','T96']:
    records=read(ROOT/panel/'requests.jsonl')
    groups=defaultdict(list)
    for r in records:groups[r['id'],r['arm']].append(r)
    first={k:next(r for r in vals if r['repeat']==0) for k,vals in groups.items()}
    ids=sorted({q for q,a in first});comparisons={};changes={}
    for name,a,b in [('D-A','bf16_serial','bf16_batch4'),('C-B','fp32_serial','fp32_batch4'),('C-A','bf16_serial','fp32_batch4'),('B-A','bf16_serial','fp32_serial')]:
        counts=Counter();absolute=[]
        for q in ids:
            x,y=first[q,a],first[q,b]
            counts.update(full_ranking=x['order']!=y['order'],top10_order=x['order'][:10]!=y['order'][:10],
                          selected_set=set(x['selected'])!=set(y['selected']),context=x['context_sha256']!=y['context_sha256'],
                          output=x['output_sha256']!=y['output_sha256'])
            changes[q,name]=x['context_sha256']!=y['context_sha256']
            absolute.extend(abs(v-w) for v,w in zip(x['scores'],y['scores']))
        a_sum=sum(statistics.median(r['seconds']['total'] for r in groups[q,a]) for q in ids)
        b_sum=sum(statistics.median(r['seconds']['total'] for r in groups[q,b]) for q in ids)
        comparisons[name]={**dict(counts),'mean_absolute_score_difference':statistics.mean(absolute),
          'max_absolute_score_difference':max(absolute),'request_time_saving':1-b_sum/a_sum}
    cells=Counter({'both':0,'native_only':0,'fp32_only':0,'neither':0})
    for q in ids:
        n,p=changes[q,'D-A'],changes[q,'C-B']
        cells['both' if n and p else 'native_only' if n else 'fp32_only' if p else 'neither']+=1
    b,c=cells['native_only'],cells['fp32_only'];n=b+c
    p=sum(math.comb(n,k) for k in range(b,n+1))/2**n
    pairs=read(ROOT/panel/'question_metrics.jsonl')
    means={a:{m:statistics.mean(r['metrics'][a][m] for r in pairs) for m in ['f1','em']} for a in ['bf16_serial','bf16_batch4','fp32_serial','fp32_batch4']}
    out[panel]={'questions':len(ids),'requests':len(records),'comparisons':comparisons,'contingency':dict(cells),'exact_p':p,'quality_means':means}
expected=json.loads((ROOT/'expected_primary.json').read_text())
for panel in out:
    assert out[panel]['contingency']==expected[panel]['contingency']
    assert math.isclose(out[panel]['exact_p'],expected[panel]['exact_p'],rel_tol=1e-12)
(ROOT/'replay_result.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'state':'passed','panels':out},indent=2))
