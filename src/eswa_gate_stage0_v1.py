"""CPU provenance audit; fail closed before any gating simulation."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import collections
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / 'results/eswa_supplement_v2'
OUT = ROOT / 'results/eswa_gate_20261110_v1'
SEED = '20261109'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def key(s):return hashlib.sha256((SEED+'|'+s).encode()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x]
def norm(s):return ' '.join(re.findall(r'\w+',s.lower()))
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(path, value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def table(path, data):
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader()
        for row in data:
            w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def independently_rank(question, docs):
    tokens=[collections.Counter(re.findall(r'\w+',d['text'].lower())) for d in docs]
    dl=[sum(t.values()) for t in tokens];avg=sum(dl)/len(dl)
    df=collections.Counter(w for c in tokens for w in c)
    query=set(re.findall(r'\w+',question.lower()))
    scores=[sum(math.log1p((len(docs)-df[t]+.5)/(df[t]+.5))*c[t]*2.5/(c[t]+1.5*(.25+.75*n/max(avg,1)))
                for t in query if c[t]) for c,n in zip(tokens,dl)]
    return sorted(range(len(docs)),key=lambda i:(-scores[i],i))[:100],scores


def pack(documents, order, tok):
    chosen=[];text=''
    for i in order[:10]:
        proposed='\n\n'.join('Title: '+documents[j]['title']+'\n'+documents[j]['text'] for j in chosen+[i])
        if len(tok.encode(proposed,add_special_tokens=False))<=512:
            chosen.append(i);text=proposed
    return {'selected':chosen,'context':text,'context_tokens':len(tok.encode(text,add_special_tokens=False))}


def main():
    assert os.name!='nt','Remote CPU only'
    assert not (ROOT/'logs/research.pause').exists()
    assert not OUT.exists(),'Do not overwrite a prior stage run'
    OUT.mkdir();base=OUT/'stage0';base.mkdir()
    result={'stage':0,'started_utc':utc(),'state':'running','errors':[]}
    try:
        frozen=load(V2/'freeze.json')
        for name,value in frozen['source_sha256'].items():assert sha(ROOT/name)==value,('frozen source hash',name)
        assert sha(V2/'inputs.jsonl')==frozen['inputs_sha256']
        assert sha(V2/'schedule.jsonl')==frozen['schedule_sha256']
        original=rows(V2/'inputs.jsonl');inputs={r['id']:r for r in original}
        assert len(inputs)==len(original)==256
        qas=[r for r in original if r['dataset']=='qasper'];hot=[r for r in original if r['dataset']=='hotpot']
        assert len(qas)==192 and len(hot)==64
        assert len({r['paper_id'] for r in qas})==192
        assert [r['id'] for r in qas]==[r['id'] for r in sorted(qas,key=lambda r:key('order|'+r['id']))]
        assert not {r['id'] for r in qas}&{r['id'] for r in hot}
        assert not {norm(r['question']) for r in qas}&{norm(r['question']) for r in hot}
        raw=load(ROOT/'data/eswa_supplement_v2/qasper-dev-v0.3.json')
        eligible=[]
        for pid,paper in raw.items():
            docs=[{'title':sec['section_name'] or paper['title'],'text':text,'paper_id':pid,'paragraph_id':f'{si}:{pi}'}
                  for si,sec in enumerate(paper['full_text']) for pi,text in enumerate(sec['paragraphs']) if text.strip()]
            questions=[q for q in paper['qas'] if q['question'].strip()]
            if len(docs)>=10 and questions:eligible.append((pid,docs,questions))
        eligible=sorted(eligible,key=lambda r:key('paper|'+r[0]))[:192]
        rebuilt={};bm25_max_error=0.
        for pid,docs,questions in eligible:
            question=min(questions,key=lambda r:key('question|'+r['question_id']))
            qid=question['question_id'];assert qid in inputs
            row=inputs[qid];assert row['paper_id']==pid and row['question']==question['question']
            order,scores=independently_rank(question['question'],docs)
            assert row['documents']==[docs[i] for i in order],('QASPER candidate mapping',qid)
            bm25_max_error=max(bm25_max_error,max(abs(x-y) for x,y in zip(row['bm25_scores'],[scores[i] for i in order])))
            assert bm25_max_error<1e-10
            rebuilt[qid]=row
        assert set(rebuilt)=={r['id'] for r in qas}
        oldhot={r['id']:r for r in rows(ROOT/'results/prospective_stage22_reserved64/inputs.jsonl')}
        for r in hot:assert r['question']==oldhot[r['id']]['question'] and r['documents']==oldhot[r['id']]['documents']
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen--Qwen3-8B'),local_files_only=True)
        qtok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen--Qwen3-Reranker-0.6B'),local_files_only=True)
        mtok=AutoTokenizer.from_pretrained(str(ROOT/'models/cross-encoder--ms-marco-MiniLM-L-6-v2'),local_files_only=True)
        prefix=qtok.encode('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n',add_special_tokens=False)
        suffix=qtok.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',add_special_tokens=False)
        per_question=[];changed={};counts={};audited=0;encoded_audited=0
        saved_summary=load(V2/'analysis/summary.json')
        for model in ['qwen','minilm']:
            path=V2/f'scores_{model}.jsonl';runtime=load(V2/f'score_runtime_{model}.json')
            assert runtime['state']=='complete' and sha(path)==runtime['scores_sha256']
            recs=rows(path);mapping={(r['id'],r['arm'],r['rep']):r for r in recs}
            assert len(recs)==len(mapping)==1216
            scheduled=[r for r in rows(V2/'schedule.jsonl') if r['model']==model]
            assert set(mapping)=={(r['id'],r['arm'],r['rep']) for r in scheduled}
            for r in recs:
                assert r['model']==model and r['id'] in inputs and r['dataset']==inputs[r['id']]['dataset']
                assert len(r['scores'])==len(inputs[r['id']]['documents'])
                assert all(math.isfinite(x) for x in r['scores'])
                assert sorted(range(len(r['scores'])),key=lambda i:(-r['scores'][i],i))==r['order']
            changed[model]=[]
            for row in original:
                qid=row['id'];dataset=row['dataset']
                serial=mapping[qid,'bf16_serial',0];batch=mapping[qid,'bf16_batch4',0]
                enc=[]
                for doc in row['documents']:
                    text='Title: '+doc['title']+'\n'+doc['text']
                    if model=='qwen':
                        instruction=f"<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: {row['question']}\n<Document>: {text}"
                        ids=qtok.encode(instruction,add_special_tokens=False)[:1024-len(prefix)-len(suffix)]
                        enc.append({'input_ids':prefix+ids+suffix})
                    else:enc.append(dict(mtok(row['question'],text,truncation=True,max_length=512)))
                assert digest(enc)==serial['encoded_sha256']==batch['encoded_sha256'],('input identity',model,qid)
                encoded_audited+=1
                left=pack(row['documents'],serial['order'],tok);right=pack(row['documents'],batch['order'],tok)
                assert left==serial['packs']['512'] and right==batch['packs']['512'],('pack',model,qid)
                audited+=2
                context_diff=left['context']!=right['context'];set_diff=set(left['selected'])!=set(right['selected'])
                c=counts.setdefault((model,dataset),{'n':0,'context_changes':0,'set_changes':0})
                c['n']+=1;c['context_changes']+=context_diff;c['set_changes']+=set_diff
                record={'model':model,'dataset':dataset,'question_id':qid,'paper_id':row.get('paper_id',''),
                        'question':row['question'],'serial_selected_indices':left['selected'],'batch4_selected_indices':right['selected'],
                        'context_equal':not context_diff,'selected_members_equal':not set_diff,
                        'serial_context_sha256':hashlib.sha256(left['context'].encode()).hexdigest(),
                        'batch4_context_sha256':hashlib.sha256(right['context'].encode()).hexdigest(),
                        'serial_context_tokens':left['context_tokens'],'batch4_context_tokens':right['context_tokens'],
                        'candidate_count':len(row['documents']),'encoded_inputs_sha256':digest(enc),
                        'source_input_row_sha256':digest(row),'serial_record_sha256':digest(serial),'batch4_record_sha256':digest(batch)}
                per_question.append(record)
                if dataset=='qasper' and context_diff:
                    changed[model].append({**record,'serial_context':left['context'],'batch4_context':right['context'],
                                          'serial_selected_source_paragraphs':[row['documents'][i]['paragraph_id'] for i in left['selected']],
                                          'batch4_selected_source_paragraphs':[row['documents'][i]['paragraph_id'] for i in right['selected']]})
            for dataset in ['qasper','hotpot']:
                s=next(x for x in saved_summary['cells'] if x['model']==model and x['dataset']==dataset and x['budget']==512 and x['condition']=='native')
                c=counts[model,dataset]
                assert c['n']==s['n'] and c['context_changes']==s['context_changes'] and c['set_changes']==s['selected_set_changes'],('summary mismatch',model,dataset)
        assert len(changed['qwen'])==51 and len(changed['minilm'])==9
        table(base/'all_questions_audit.csv',per_question)
        for model,data in changed.items():
            table(base/f'qasper_{model}_changed_contexts.csv',data)
            with (base/f'qasper_{model}_changed_contexts.jsonl').open('x',encoding='utf-8') as f:
                for row in data:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        result.update(state='passed',passed=True,stop=False,finished_utc=utc(),
            counts=[{'model':m,'dataset':d,**v} for (m,d),v in counts.items()],
            independent_packs_rebuilt=audited,tokenization_inputs_rebuilt=encoded_audited,
            qasper_questions_rebuilt_from_official_source=len(rebuilt),bm25_max_absolute_reconstruction_error=bm25_max_error,
            cross_dataset_id_overlap=0,cross_dataset_normalized_question_overlap=0,
            qasper_unique_papers=len({r['paper_id'] for r in qas}),
            qasper_ordered_ids=[r['id'] for r in qas],
            interpretation='Equal fractions are reproduced from distinct QASPER question/candidate/score/context records, not from duplicated Hotpot counts.',
            source_hashes={'v2_inputs':sha(V2/'inputs.jsonl'),'qasper_original':sha(ROOT/'data/eswa_supplement_v2/qasper-dev-v0.3.json'),
                           'v2_freeze':sha(V2/'freeze.json'),'stage0_code':sha(__file__)})
    except BaseException as exc:
        result.update(state='failed',passed=False,stop=True,finished_utc=utc())
        result['errors'].append(repr(exc))
        raise
    finally:
        dump(base/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='qasper_ordered_ids'},ensure_ascii=False))


if __name__=='__main__':main()
