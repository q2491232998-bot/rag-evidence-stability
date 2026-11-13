"""Post-result development diagnosis explicitly authorized by the user."""
import csv
import itertools
import hashlib
import json
from eswa_gate_stage1_v1 import ROOT,V2,OUT,Packer,ranking,load,rows,table,dump,sha,utc


def main():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen--Qwen3-8B'),local_files_only=True)
    base=OUT/'five_case_diagnosis';base.mkdir(exist_ok=False)
    inputs={r['id']:r for r in rows(V2/'inputs.jsonl')}
    scores={(r['id'],r['arm']):r for r in rows(V2/'scores_qwen.jsonl') if r['rep']==0}
    with (OUT/'stage1/qwen_selected_tau_test_failures.csv').open(encoding='utf-8',newline='') as f:failures=list(csv.DictReader(f))
    allpairs=[];summary=[];candidate_rows=[]
    for item in failures:
        qid=item['question_id'];row=inputs[qid]
        a=scores[qid,'bf16_serial']['scores'];b=scores[qid,'bf16_batch4']['scores']
        marked=json.loads(item['rescored_indices']);h=[a[i] if i in marked else b[i] for i in range(len(a))]
        order=ranking(h);serial_order=ranking(a);batch_order=ranking(b)
        pos={idx:p for p,idx in enumerate(order)};spos={idx:p for p,idx in enumerate(serial_order)};bpos={idx:p for p,idx in enumerate(batch_order)}
        pack=Packer(row['documents'],tok);current=pack(order);target=pack(serial_order)
        assert current['selected']==json.loads(item['selected_indices'])
        relevant=[]
        for left,right in itertools.combinations(order,2):
            if spos[left]<spos[right]:continue
            swapped=list(order);swapped[pos[left]],swapped[pos[right]]=right,left
            effect=pack(swapped)['context']!=current['context']
            if not effect:continue
            c=list(h);c[left]=a[left];c[right]=a[right];after=pack(ranking(c))
            reasons=[]
            if max(pos[left],pos[right])>=12:reasons.append('outside_top12')
            if pos[right]-pos[left]>1:reasons.append('nonadjacent')
            if h[left]-h[right]>=.25:reasons.append('gap_ge_tau')
            if left in marked and right in marked:reasons.append('already_serial')
            rec={'question_id':qid,'left_index':left,'right_index':right,'hybrid_rank_left':pos[left]+1,'hybrid_rank_right':pos[right]+1,
                 'hybrid_gap':h[left]-h[right],'serial_score_left':a[left],'serial_score_right':a[right],
                 'batch4_score_left':b[left],'batch4_score_right':b[right],'serial_rank_left':spos[left]+1,'serial_rank_right':spos[right]+1,
                 'swap_changes_context':effect,'swap_restores_serial':pack(swapped)['context']==target['context'],
                 'rescore_pair_restores_serial':after['context']==target['context'],'miss_conditions':reasons}
            relevant.append(rec);allpairs.append(rec)
        for i,d in enumerate(row['documents']):
            candidate_rows.append({'question_id':qid,'index':i,'title':d['title'],'paragraph_id':d.get('paragraph_id',''),
                                   'serial_score':a[i],'batch4_score':b[i],'hybrid_score':h[i],
                                   'serial_rank':spos[i]+1,'batch4_rank':bpos[i]+1,'hybrid_rank':pos[i]+1,
                                   'serial_selected':i in target['selected'],'hybrid_selected':i in current['selected'],'rescored_by_v1':i in marked,
                                   'paragraph_tokens':len(tok.encode(pack.blocks[i],add_special_tokens=False))})
        summary.append({'question_id':qid,'question':row['question'],'serial_selected':target['selected'],'v1_selected':current['selected'],
                        'v1_rescored':marked,'relevant_inverted_pairs':relevant,
                        'serial_top10':serial_order[:10],'v1_top15':order[:15]})
    table(base/'candidate_scores_ranks.csv',candidate_rows);table(base/'missed_pairs.csv',allpairs)
    dump(base/'diagnosis.json',{'utc':utc(),'status':'posthoc_development_only','old_128_no_longer_test':True,'cases':summary,
                              'stage1_decision_sha256':sha(OUT/'stage1/decision.json'),'code_sha256':sha(__file__)})
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
