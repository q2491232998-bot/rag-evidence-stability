"""Remote CPU source/mapping audit while the existing GPU queue waits; no model scores."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv
import io
import json
import re
import tarfile
from eswa_stage2_common import *
from eswa_gate_stage0_v1 import independently_rank
from eswa_stage2_metrics import hotpot_gold


def norm(text):return ' '.join(re.findall(r'\w+',text.lower()))


def main():
    check();out=BASE/'input_audit';out.mkdir(exist_ok=False)
    result={'utc':utc(),'state':'running','scoring_performed':False,'gate_protocol_modified':False}
    per_question=[]
    try:
        lock=load(BASE/'definition_lock.json')
        for path,want in lock['sha256'].items():assert sha(ROOT/path)==want,('definition_changed',path)
        sampling=load(BASE/'sampling.json');acq=load(DATA/'acquisition.json')
        assert sha(DATA/'acquisition.json')==sampling['source_acquisition_sha256']
        for entry in acq['datasets']:assert sha(DATA/entry['file'])==entry['sha256']
        historical_ids=set();historical_papers=set();historical_texts=set()
        for entry in sampling['history_files']:
            assert sha(ROOT/entry['path'])==entry['sha256'],('historical_input_changed',entry['path'])
            count=0
            for row in read_jsonl(ROOT/entry['path']):
                q=row.get('question');qid=row.get('id',row.get('_id',row.get('question_id')))
                if isinstance(q,str) and qid is not None:
                    historical_ids.add(str(qid));historical_texts.add(norm(q));count+=1
                    if row.get('paper_id'):historical_papers.add(str(row['paper_id']))
            assert count==entry['rows']
        historical_papers.update(r['paper_id'] for r in read_jsonl(ROOT/'results/eswa_supplement_v2/inputs.jsonl') if r['dataset']=='qasper')
        assert len(historical_ids)==sampling['historical_ids'] and len(historical_papers)==sampling['historical_papers']
        with tarfile.open(DATA/'qasper-test-and-evaluator-v0.3.tgz') as archive:
            raw=json.load(archive.extractfile('qasper-test-v0.3.json'))
        assert raw==load(DATA/'qasper-test-v0.3.json')
        qrows=read_jsonl(BASE/'qasper_inputs.jsonl');qlabels=read_jsonl(DATA/'qasper_labels.jsonl')
        labels={r['id']:r for r in qlabels};assert len(labels)==len(qlabels)==len(qrows)==256
        qids={r['id'] for r in qrows};papers={r['paper_id'] for r in qrows}
        assert len(qids)==len(papers)==256 and not papers&historical_papers and not qids&historical_ids
        assert [r['id'] for r in qrows if r['role']=='bge_calibration']==sampling['calibration_ids']
        assert [r['id'] for r in qrows if r['role']=='evaluation']==sampling['qasper_evaluation_ids']
        maximum_error=0.;generic=0;eval_generic=0;cal_generic=0
        for row in qrows:
            paper=raw[row['paper_id']];qas=[q for q in paper['qas'] if q['question_id']==row['id']];assert len(qas)==1
            qa=qas[0];assert qa['question']==row['question'] and labels[row['id']]['qas']==qa
            assert labels[row['id']]['paper_id']==row['paper_id'] and labels[row['id']]['role']==row['role']
            docs=[{'title':section['section_name'] or paper['title'],'text':text,'paper_id':row['paper_id'],'paragraph_id':f'{si}:{pi}'}
                  for si,section in enumerate(paper['full_text']) for pi,text in enumerate(section['paragraphs']) if text.strip()]
            ranking,scores=independently_rank(qa['question'],docs)
            assert row['documents']==[docs[i] for i in ranking],('candidate_mapping',row['id'])
            error=max(abs(a-scores[i]) for a,i in zip(row['bm25_scores'],ranking));maximum_error=max(maximum_error,error)
            assert error<1e-10,('bm25_numeric_error',row['id'],error)
            seen=norm(qa['question']) in historical_texts;generic+=seen
            if row['role']=='evaluation':eval_generic+=seen
            else:cal_generic+=seen
            per_question.append({'id':row['id'],'dataset':'qasper','role':row['role'],'paper_id':row['paper_id'],
                                 'source_question_and_labels_exact':True,'candidate_paragraph_mapping_exact':True,
                                 'candidate_count':len(row['documents']),'bm25_max_abs_rebuild_error':error,
                                 'historical_id_overlap':False,'historical_paper_overlap':False,'generic_question_text_seen':seen,
                                 'support_facts_total':'','unresolvable_support_facts':''})
        assert generic==sampling['qasper_generic_question_text_overlap']
        full_hot=load(DATA/'hotpot_dev_distractor_v1.json');hotmap={r['_id']:r for r in full_hot};assert len(full_hot)==len(hotmap)==7405
        hqueries=read_jsonl(DATA/'hotpot_queries.jsonl');hlabels=read_jsonl(DATA/'hotpot_labels.jsonl')
        assert len(hqueries)==len(hlabels)==200
        assert [r['id'] for r in hqueries]==[r['_id'] for r in hlabels]==sampling['hotpot_evaluation_ids']
        hids={r['id'] for r in hqueries};assert len(hids)==200 and not hids&historical_ids and not hids&qids
        unresolvable=0;support_count=0
        for query,label in zip(hqueries,hlabels):
            assert label==hotmap[query['id']] and query['question']==label['question']
            assert norm(query['question']) not in historical_texts
            facts=hotpot_gold(label);missing=sum(not f['resolvable'] for f in facts);unresolvable+=missing;support_count+=len(facts)
            per_question.append({'id':query['id'],'dataset':'hotpot','role':'evaluation','paper_id':'',
                                 'source_question_and_labels_exact':True,'candidate_paragraph_mapping_exact':'pending_gpu_retrieval',
                                 'candidate_count':'','bm25_max_abs_rebuild_error':'',
                                 'historical_id_overlap':False,'historical_paper_overlap':'','generic_question_text_seen':False,
                                 'support_facts_total':len(facts),'unresolvable_support_facts':missing})
        assert len(per_question)==456
        result.update(state='passed',qasper_questions=256,qasper_evaluation=192,bge_calibration=64,hotpot_dev_questions=200,
                      qasper_new_papers=256,qasper_bm25_max_abs_rebuild_error=maximum_error,
                      historical_inputs_unchanged=len(sampling['history_files']),historical_id_overlap=0,historical_paper_overlap=0,
                      qasper_generic_text_overlap_evaluation=eval_generic,qasper_generic_text_overlap_calibration=cal_generic,
                      hotpot_support_facts=support_count,hotpot_unresolvable_support_facts=unresolvable,
                      hotpot_candidates_pending_gpu=True,source_sha256=sha(__file__),definition_lock_sha256=sha(BASE/'definition_lock.json'))
    except BaseException as exc:
        result.update(state='failed',error=repr(exc));raise
    finally:
        if per_question:
            with (out/'per_question_source_audit.csv').open('x',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(per_question[0]));writer.writeheader();writer.writerows(per_question)
        save(out/'audit.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
