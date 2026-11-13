"""Independent remote CPU replay, all configurations, Chinese report and auditable CSV."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,csv,importlib.util,json,statistics,shutil
from eswa_4090_common_v1 import *
from eswa_stage2_metrics import *

def mean(xs):
    xs=[x for x in xs if x is not None];return sum(xs)/len(xs) if xs else None
def csvwrite(path,rows):
    rows=list(rows);fields=list(dict.fromkeys(k for r in rows for k in r)) or ['no_records']
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict,tuple)) else v for k,v in r.items()})
def independent_pack(row,ranking,tok):
    chosen=[];text=''
    for i in ranking[:10]:
        d=row['documents'][i];block='Title: '+d['title']+'\n'+d['text'];proposal=text+'\n\n'+block if chosen else block
        if len(tok.encode(proposal,add_special_tokens=False))<=512:chosen.append(i);text=proposal
    return chosen,text,len(tok.encode(text,add_special_tokens=False))

def main():
    verify();assert load(OUT/'reader_runtime.json')['state']=='complete'
    from transformers import AutoTokenizer
    from run_eswa_stage2_v1 import Packer
    from eswa_gate_methods_v2 import flagged,order
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen--Qwen3-8B'),local_files_only=True)
    official_path=DATA/'qasper_evaluator.py';spec=importlib.util.spec_from_file_location('official_qasper_4090',official_path);official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    raw=load(DATA/'qasper_dev_selected.json')|load(DATA/'qasper_test_selected.json');refs=official.get_answers_and_evidence(raw,text_evidence_only=True)
    rows={r['id']:r for r in read_jsonl(OUT/'inputs.jsonl')};crossrows={r['id']:r for r in read_jsonl(DATA/'cross_hardware_inputs.jsonl')};hot={r['_id']:r for r in read_jsonl(DATA/'hotpot_labels.jsonl')}
    answers={r['key']:r for r in read_jsonl(OUT/'answers.jsonl')};lr=read_jsonl(OUT/'reader_links.jsonl');links={identity(r):r['key'] for r in lr};assert len(lr)==len(links)
    assert sha(OUT/'answers.jsonl')==load(OUT/'reader_runtime.json')['answers_sha256']
    all_records={};byphase={};checks=collections.Counter();failures=[];eval_exports=[];cal_exports=[];cross_exports=[];error_exports=[];cache={};replays={};runtime_exports=[]
    for phase in ['calibration','evaluation','cross']:
        rr=[]
        for model in ['qwen','bge']:
            p=OUT/f'{phase}_{model}.jsonl';rt=load(OUT/f'runtime_{phase}_{model}.json');assert rt['state']=='complete' and rt['parameters_unchanged'] and rt['buffers_unchanged'] and sha(p)==rt['records_sha256']
            runtime_exports.append({k:rt[k] for k in ['model','phase','state','utc','finished_utc','seconds','hardware','records','parameter_hash','parameters_unchanged','buffers_unchanged','peak_allocated_bytes']})
            records=read_jsonl(p);expected={identity(t) for t in tasks(model,phase)};assert len(records)==len(expected) and {identity(r) for r in records}==expected;rr+=records
        byphase[phase]=rr;all_records.update({identity(r):r for r in rr})
    assert len(all_records)==26896 and len(links)==len(byphase['evaluation'])
    baseline={identity(r):r for r in read_jsonl(DATA/'baseline_3090ti_serial.jsonl')}
    medians={}
    for phase in ['calibration','evaluation']:
        for r in byphase[phase]:
            k=identity(r)[:4]
            if k not in medians:medians[k]=statistics.median(all_records[k+(rep,None)]['total_seconds'] for rep in range(2))
    encoded={};rescore_errors=[]
    for phase in ['calibration','evaluation','cross']:
        for r in byphase[phase]:
            row=(crossrows if phase=='cross' else rows)[r['id']];k=identity(r);reference=all_records[r['model'],r['id'],'serial',1,r['rep'],None]
            assert r['candidate_count']==len(row['documents'])==len(r['scores'])==len(r['initial_scores'])
            qk=r['model'],r['id'];encoded.setdefault(qk,r['encoded_sha256']);assert encoded[qk]==r['encoded_sha256']
            work=list(r['initial_scores']);marked=set();replay_key=(r['model'],r['id'],r['method'],tuple(work),r['epsilon'],digest(r['rounds']))
            if replay_key not in replays:
                packer=Packer(row['documents'],tok);band=None
                for iteration,rd in enumerate(r['rounds']):
                    indices=rd['indices'];assert not marked&set(indices)
                    if r['method']=='gate_v2':
                        fresh,pairs=flagged(work,marked,packer,r['tau'],'v2');assert fresh==indices and pairs==rd['trigger_pairs']
                    elif r['method']=='top15':assert iteration==0 and indices==sorted(range(len(work)),key=lambda i:(-work[i],i))[:15]
                    elif r['method'].startswith('band_'):
                        ranking=sorted(range(len(work)),key=lambda i:(-work[i],i));kth=work[ranking[min(10,len(work))-1]]
                        band=[i for i,s in enumerate(work) if s>=kth-2*r['epsilon']];assert iteration==0 and indices==band and r['band_indices']==band and r['batch_kth_score']==kth
                    else:raise AssertionError(('Unexpected rescoring',r['method']))
                    actual={int(i):s for i,s in rd['actual_serial_scores'].items()};assert set(indices)==set(actual)
                    for i,s in actual.items():work[i]=s
                    marked.update(indices)
                if r['method']=='gate_v2':assert not flagged(work,marked,packer,r['tau'],'v2')[0]
                ranking=sorted(band if band is not None else range(len(work)),key=lambda i:(-work[i],i))
                if band is not None:ranking=ranking[:10]
                assert ranking==r['order'] and work==r['scores'] and sorted(marked)==r['rescored_indices'] and len(marked)==r['rescored_count']
                assert r['rescored_fraction']==len(marked)/len(work);replays[replay_key]=(list(work),list(ranking),sorted(marked),band)
            expected_scores,expected_ranking,expected_marked,expected_band=replays[replay_key]
            assert r['scores']==expected_scores and r['order']==expected_ranking and r['rescored_indices']==expected_marked
            assert r['rescored_count']==len(expected_marked) and r['rescored_fraction']==len(expected_marked)/len(expected_scores)
            if r['method'].startswith('band_'):assert r['band_indices']==expected_band
            for rd in r['rounds']:
                for i,s in rd['actual_serial_scores'].items():
                    error=abs(s-reference['scores'][int(i)]);checks['rescored_values_checked']+=1
                    if error:
                        checks['rescored_values_different']+=1;rescore_errors.append({k:r[k] for k in ['model','id','method','batch_size','rep']}|{'candidate_index':int(i),'error':error,'actual_serial_rescore':s,'reference_serial_score':reference['scores'][int(i)]})
            pk=row['id'],tuple(r['order'][:10])
            if pk not in cache:cache[pk]=independent_pack(row,r['order'],tok)
            selected,text,tokens=cache[pk];assert selected==r['selected'] and text==r['context'] and tokens==r['tokens'];checks['independent_pack_records']+=1
            assert r['total_seconds']>0 and r['gpu_isolation_observed']
            export={k:r[k] for k in ['model','id','dataset','paper_id','method','batch_size','rep','role','tau','epsilon','candidate_count','selected','rescored_indices','rescored_count','rescored_fraction','total_seconds','forward_seconds','gate_check_seconds','forward_calls','gpu_isolation_observed']}
            export.update(question=row['question'],reference_hardware='RTX4090',source_split=row.get('source_split','stage2'),serial_selected=reference['selected'],context_equal_serial=text==reference['context'],members_equal_serial=set(selected)==set(reference['selected']),context_sha256=digest(text),serial_context_sha256=digest(reference['context']),context_tokens=tokens,
                          actual_rescoring_values_match_serial=all(s==reference['scores'][int(i)] for rd in r['rounds'] for i,s in rd['actual_serial_scores'].items()))
            if phase=='cross':
                previous=[baseline[r['model'],r['id'],'serial',1,rep,None] for rep in range(2)]
                assert all(p['encoded_sha256']==r['encoded_sha256'] for p in previous)
                export.update(reference_hardware='RTX3090Ti_rep0',serial_selected=previous[0]['selected'],context_equal_serial=text==previous[0]['context'],members_equal_serial=set(selected)==set(previous[0]['selected']),serial_context_sha256=digest(previous[0]['context']))
                export.update(selected_3090ti_rep0=previous[0]['selected'],selected_3090ti_rep1=previous[1]['selected'],context_equal_3090ti_rep0=text==previous[0]['context'],context_equal_3090ti_rep1=text==previous[1]['context'],members_equal_3090ti_rep0=set(selected)==set(previous[0]['selected']),baseline_3090ti_repeat_context_identical=previous[0]['context']==previous[1]['context'],max_score_difference_3090ti_rep0=max(abs(a-b) for a,b in zip(r['scores'],previous[0]['scores'])))
                cross_exports.append(export);continue
            other=all_records[r['model'],r['id'],r['method'],r['batch_size'],1-r['rep'],None];serialtime=medians[r['model'],r['id'],'serial',1];mtime=medians[identity(r)[:4]]
            export.update(repeat_context_identical=text==other['context'],repeat_scores_identical=r['scores']==other['scores'],two_repeat_median_seconds=mtime,serial_two_repeat_median_seconds=serialtime,paired_time_ratio_to_serial=mtime/serialtime,slower_than_serial=mtime>serialtime)
            errors=[abs(b-s) for b,s in zip(r['initial_scores'],reference['scores'])];maxerr=max(errors);eps=r['epsilon'] if r['epsilon'] is not None else C['transferred_epsilons'][r['model']]
            export.update(max_abs_batch_serial_error=maxerr,error_comparison_epsilon=eps,candidates_exceeding_epsilon=sum(e>eps for e in errors),question_exceeds_epsilon=any(e>eps for e in errors),candidates_equal_epsilon=sum(e==eps for e in errors))
            docs=[row['documents'][i] for i in selected]
            metric=qasper_evidence(docs,refs[row['id']],official) if row['dataset']=='qasper' else hotpot_evidence(docs,hot[row['id']])
            export.update(metric)
            if phase=='calibration':cal_exports.append(export);continue
            pred=answers[links[identity(r)]]
            if row['dataset']=='qasper':
                score={'answer_f1':max(official.token_f1_score(pred['answer'],a['answer']) for a in refs[row['id']]),'answer_em':max(float(official.normalize_answer(pred['answer'])==official.normalize_answer(a['answer'])) for a in refs[row['id']])}
            else:score=hotpot_answer(pred['answer'],hot[row['id']]['answer'])
            export.update(score,reader_key=pred['key'],answer=pred['answer'],reader_cap_hit=pred['cap_hit']);eval_exports.append(export)
            if r['method'].startswith('band_'):
                for i,e in enumerate(errors):
                    if e>eps:error_exports.append({k:r[k] for k in ['model','id','dataset','method','batch_size','rep','epsilon']}|{'candidate_index':i,'initial_batch_score':r['initial_scores'][i],'serial_score':reference['scores'][i],'absolute_error':e})
    assert len(eval_exports)==25088 and len(cal_exports)==1024 and len(cross_exports)==784
    groups=collections.defaultdict(list)
    for r in eval_exports:groups[r['model'],r['dataset'],r['method'],r['batch_size'],r['rep']].append(r)
    summary=[]
    for (model,dataset,method,size,rep),rr in sorted(groups.items()):
        ref=groups[model,dataset,'serial',1,rep];diff=sum(not r['context_equal_serial'] for r in rr)
        summary.append({'model':model,'dataset':dataset,'method':method,'batch_size':size,'rep':rep,'questions':len(rr),'context_differences':diff,'member_differences':sum(not r['members_equal_serial'] for r in rr),'rescored_fraction_mean':mean([r['rescored_fraction'] for r in rr]),'rescored_fraction_max':max(r['rescored_fraction'] for r in rr),'gold_recall_n':sum(r['gold_evidence_recall'] is not None for r in rr),'gold_recall_mean':mean([r['gold_evidence_recall'] for r in rr]),'gold_recall_delta_to_serial':mean([r['gold_evidence_recall'] for r in rr])-mean([r['gold_evidence_recall'] for r in ref]),'answer_f1_mean':mean([r['answer_f1'] for r in rr]),'answer_f1_delta_to_serial':mean([r['answer_f1'] for r in rr])-mean([r['answer_f1'] for r in ref]),'time_ratio_of_means':mean([r['two_repeat_median_seconds'] for r in rr])/mean([r['serial_two_repeat_median_seconds'] for r in rr]),'paired_time_ratio_mean':mean([r['paired_time_ratio_to_serial'] for r in rr]),'paired_time_ratio_median':statistics.median(r['paired_time_ratio_to_serial'] for r in rr),'slower_than_serial_questions':sum(r['slower_than_serial'] for r in rr),'max_abs_batch_serial_error':max(r['max_abs_batch_serial_error'] for r in rr),'epsilon':rr[0]['epsilon'],'candidates_exceeding_epsilon':sum(r['candidates_exceeding_epsilon'] for r in rr),'questions_exceeding_epsilon':sum(r['question_exceeds_epsilon'] for r in rr),'candidates_equal_epsilon':sum(r['candidates_equal_epsilon'] for r in rr),'repeat_context_disagreements':sum(not r['repeat_context_identical'] for r in rr),'zero_event_onesided95_upper':1-.05**(1/len(rr)) if diff==0 else None})
    crosssummary=[]
    for model in ['qwen','bge']:
        for dataset in ['hotpot','qasper']:
            rr=[r for r in cross_exports if r['model']==model and r['dataset']==dataset]
            crosssummary.append({'model':model,'dataset':dataset,'questions':len(rr),'context_differences_vs_3090ti_rep0':sum(not r['context_equal_3090ti_rep0'] for r in rr),'context_differences_vs_3090ti_rep1':sum(not r['context_equal_3090ti_rep1'] for r in rr),'member_differences_vs_3090ti_rep0':sum(not r['members_equal_3090ti_rep0'] for r in rr),'baseline_repeat_context_disagreements':sum(not r['baseline_3090ti_repeat_context_identical'] for r in rr),'max_score_difference':max(r['max_score_difference_3090ti_rep0'] for r in rr)})
    occupancy=read_jsonl(OUT/'gpu_occupancy.jsonl');foreign={p['pid'] for r in occupancy for p in r['processes'] if not p['own']};speech={p['pid'] for r in occupancy for p in r['processes'] if not p['own'] and p['speech_service_detected']}
    foreign_during={p['pid'] for r in occupancy if r['request'] for p in r['processes'] if not p['own']}
    invalid=read_jsonl(OUT/'contaminated_attempts.jsonl') if (OUT/'contaminated_attempts.jsonl').exists() else []
    failed=read_jsonl(OUT/'failed_attempts.jsonl') if (OUT/'failed_attempts.jsonl').exists() else []
    gpu={'sample_interval_seconds':C['gpu_monitor_interval_seconds'],'state_change_records':len(occupancy),'foreign_pids_including_queue':sorted(foreign),'speech_service_pids_including_queue':sorted(speech),'foreign_pids_during_requests':sorted(foreign_during),'contaminated_attempts':len(invalid),'failed_attempts':len(failed),'limitation':'No detection cannot exclude jobs shorter than the 0.5-second sampling interval.'}
    final=OUT/'final';final.mkdir(exist_ok=False)
    for name in ['rules_freeze.json','freeze.json','epsilon_freeze.json','epsilon_recalibrated.json','sampling.json','candidate_manifest.json']:
        shutil.copyfile(OUT/name,final/name)
    shutil.copyfile(ROOT/'research/eswa_4090_validation_protocol_v1.md',final/'protocol.md')
    csvwrite(final/'sample_manifest.csv',[{k:r.get(k,'') for k in ['id','question','paper_id','dataset','source_split','role']}|{'candidate_count':len(r['documents'])} for r in rows.values()])
    csvwrite(final/'hardware_runtime.csv',runtime_exports)
    for name,data in [('per_question.csv',eval_exports),('summary_all_repetitions.csv',summary),('band_summary.csv',[r for r in summary if r['method'].startswith('band_')]),('calibration_per_question.csv',cal_exports),('calibration_errors.csv',read_jsonl(OUT/'calibration_errors.jsonl')),('cross_hardware_per_question.csv',cross_exports),('cross_hardware_summary.csv',crosssummary),('band_exceeding_candidates.csv',error_exports),('rescoring_reference_errors.csv',rescore_errors),('contaminated_attempts.csv',[{k:r[k] for k in ['model','id','method','batch_size','rep','attempt','total_seconds','selected','rescored_count']}|{'context_sha256':digest(r['context']),'external_gpu_observations':r['external_gpu_observations']} for r in invalid]),('failed_attempts.csv',failed)]:csvwrite(final/name,data)
    save(final/'gpu_monitor_summary.json',gpu);save(final/'audit.json',{'utc':utc(),'checks':checks,'all_planned_requests_present':True,'all_saved_contexts_rebuilt':True,'all_valid_timings_observed_isolated':True,'rescoring_reference_errors':len(rescore_errors),'freeze_sha256':sha(OUT/'freeze.json'),'epsilon_freeze_sha256':sha(OUT/'epsilon_freeze.json')})
    eps=load(OUT/'epsilon_recalibrated.json');sample=load(OUT/'sampling.json');lines=['# RTX 4090 独立样本验证：中文结果报告','',f'结果生成时间：{utc()}。本轮只报告实验，不修改论文。','',
    '## 完成状态与冻结', '', '协议、输入/代码/权重及校准 ε 已按顺序冻结；两模型、全部配置、两次重复和跨硬件单次均已完成。校准先于新评估题打分。主分析沿用旧 ε，次要分析单列重新校准 ε，未根据评估结果调参。', '',
    '## 样本与来源','',f'实际样本：{json.dumps(sample["selected_counts"],ensure_ascii=False)}。历史排除 {sample["historical_ids"]} 个题 ID、{sample["historical_papers"]} 篇论文；与历史题 ID、QASPER 论文及规范化问题文本重叠均为 0。QASPER 一篇一题；本轮不同论文的问题表述重复计数为 {sample["within_new_qasper_duplicate_text_count"]}，题 ID 和论文均不同，未事后替换样本。sample_manifest.csv 保留全部问题表述及来源。HotpotQA 为官方 dev 内容，HF 固定版本传输，不声称 CMU JSON 字节身份。', '',
    '## 校准 ε（每模型一个）','', '两个数据集各 32 题、batch4/8/32、两次重复及全部候选共同取最大绝对误差，再乘 1.5。', '', '|模型|沿用 ε|校准最大误差|新 ε|','|---|---:|---:|---:|']
    for model in ['qwen','bge']:lines.append(f'|{model}|{C["transferred_epsilons"][model]}|{eps["models"][model]["maximum_absolute_error"]}|{eps["models"][model]["epsilon"]}|')
    lines+=['','## 全部主分析和次要分析结果','','下表质量为 rep0；CSV 保留 rep0/rep1 全部指标。耗时为每题两次中位数，表中耗时比为平均耗时之比；逐题比例的均值/中位数及超时题数也在 CSV。重复之间如有质量差异，紧随表格列出。','', '|模型/数据|方法/batch|context差异|重打分均值|gold召回|答案F1|耗时比|慢于serial题数|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['rep']==0:lines.append(f'|{r["model"]}/{r["dataset"]}|{r["method"]}/{r["batch_size"]}|{r["context_differences"]}/{r["questions"]}|{r["rescored_fraction_mean"]:.6f}|{r["gold_recall_mean"]:.6f}|{r["answer_f1_mean"]:.6f}|{r["time_ratio_of_means"]:.4f}|{r["slower_than_serial_questions"]}|')
    discord=[{k:r[k] for k in ['model','dataset','method','batch_size','rep','repeat_context_disagreements']} for r in summary if r['repeat_context_disagreements']]
    lines+=['',f'重复间 context 不一致配置：{json.dumps(discord,ensure_ascii=False)}。所有与 serial 的 gold/F1 增减均保留，不预设准确率相同。','', '## Band 的误差界检查','', '|模型/数据|方法/batch/rep|ε|最大误差|越界候选|越界题|','|---|---|---:|---:|---:|---:|']
    for r in summary:
        if r['method'].startswith('band_'):lines.append(f'|{r["model"]}/{r["dataset"]}|{r["method"]}/{r["batch_size"]}/{r["rep"]}|{r["epsilon"]}|{r["max_abs_batch_serial_error"]}|{r["candidates_exceeding_epsilon"]}|{r["questions_exceeding_epsilon"]}|')
    lines+=['','越界严格使用 >ε，等于边界另外计数。经验 ε 的迁移失效不因 context 碰巧一致而隐藏；只有实际误差满足界且真实重打分与参考一致时，band 的条件保证适用。','', '## 同题跨硬件 serial','', '|模型/数据|题数|相对3090Ti rep0差异|相对3090Ti rep1差异|','|---|---:|---:|---:|']
    for r in crosssummary:lines.append(f'|{r["model"]}/{r["dataset"]}|{r["questions"]}|{r["context_differences_vs_3090ti_rep0"]}|{r["context_differences_vs_3090ti_rep1"]}|')
    lines+=['','此组使用原阶段2完全相同的392道输入，不能当作新题泛化样本。所有编码哈希已核对；逐题差异、旧serial重复差异和分数差异详见CSV。','', '## GPU占用、无效尝试与局限','',f'GPU 监控摘要：{json.dumps(gpu,ensure_ascii=False)}。外部服务未被停止；检测到占用时等待。所有有效计时请求的前后及后台采样未检测到外部进程；采样间隔内极短占用仍可能漏检。', '',f'独立重放 {checks["independent_pack_records"]} 条请求；核对 {checks["rescored_values_checked"]} 个真实重打分值，其中与对应 serial 不同 {checks["rescored_values_different"]} 个。错误逐条导出，不以删除样本掩盖。', '',f'Reader：{json.dumps(load(OUT/"reader_runtime.json"),ensure_ascii=False)}。未复用3090Ti答案。','', '零差异不是总体零风险；CSV 的单侧95%样本上界为 1−0.05^(1/n)，n不乘重复次数，使用独立同分布近似。局限仍为两种GPU环境、一个reader、一种打包、两个数据集；硬件比较同时受记录的软件/驱动环境差异约束。', '', '阶段1仍记录为未达到预设90%（实际86.49%），继续研究由用户决定。本轮无事后修改阈值、方法或论文。']
    (final/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(final/'manifest.json',{'utc':utc(),'files':{p.name:sha(p) for p in sorted(final.iterdir()) if p.is_file()},'state':'complete'})
    print(json.dumps({'state':'complete','main_and_secondary_rows':len(eval_exports),'calibration_rows':len(cal_exports),'cross_rows':len(cross_exports),'final':str(final)}),flush=True)

if __name__=='__main__':main()
