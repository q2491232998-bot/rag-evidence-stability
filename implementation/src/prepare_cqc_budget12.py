import ast
import datetime
import json
import os

from common import ROOT, read_jsonl
from cqc_budget12_core import file_sha, settings, source_path, write_new


def extract_prompts():
    path = ROOT / 'references/cqc_rag_stage09/inference/main_reranker_record_trivaiqa.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    values = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ['generate_paraphrases', '_build_prompt_content']:
            for assign in ast.walk(node):
                if isinstance(assign, ast.Assign) and isinstance(assign.value, ast.Constant) and isinstance(assign.value.value, str):
                    names = [t.id for t in assign.targets if isinstance(t, ast.Name)]
                    if 'system_prompt' in names:
                        values['paraphrase_system_original'] = assign.value.value
                    if 'sys_prompt' in names:
                        values['answer_system'] = assign.value.value
    values['paraphrase_system'] = values['paraphrase_system_original'].replace('trivia question answering task (TrivialQA)', 'multi-hop question answering task (HotpotQA)').replace('generation task requires 6 Paraphrases.', 'generation task requires 4 Paraphrases.')
    path = ROOT / 'references/cqc_rag_stage09/evaluate/autoregression_evalutor.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'prompt' for t in node.targets) and isinstance(node.value, ast.JoinedStr):
            parts = []
            for part in node.value.values:
                if isinstance(part, ast.Constant):
                    parts.append(part.value)
                else:
                    assert isinstance(part, ast.FormattedValue) and isinstance(part.value, ast.Name)
                    parts.append('{' + {'context': 'evidence', 'query': 'question', 'answer': 'answer'}[part.value.id] + '}')
            values['verification_template'] = ''.join(parts)
    assert set(values) == {'paraphrase_system_original', 'paraphrase_system', 'answer_system', 'verification_template'}
    values['attribution'] = 'Prompts derived statically, without executing third-party code, from CQC-RAG commit baffe3c8b5206a6ff9d4b92db784971f181f1350; internal attributed research use, not public redistribution clearance.'
    return values


def main():
    config = settings()
    base = ROOT / config['output']
    assert not base.exists()
    prompt_path = ROOT / 'configs/cqc_budget_stage12_prompts.json'
    if prompt_path.exists():
        assert json.loads(prompt_path.read_text(encoding='utf-8')) == extract_prompts()
    else:
        temp = prompt_path.with_name(prompt_path.name + '.pending')
        temp.write_text(json.dumps(extract_prompts(), indent=2, ensure_ascii=False), encoding='utf-8')
        os.replace(temp, prompt_path)
    source = source_path(config['candidate_source'])
    candidates = read_jsonl(source)
    used = {r['id']: r['question'] for r in read_jsonl(ROOT / 'results/reader_replay_stage10/inputs.jsonl') if r['dataset'] == 'hotpot'}
    assert len(used) == len(candidates) == 128
    assert {r['id'] for r in candidates} == set(used)
    from hashlib import sha256
    selected = sorted(candidates, key=lambda r: sha256((config['sample_key'] + r['id']).encode()).hexdigest())[:config['questions']]
    rows = []
    for r in selected:
        assert r['question'] == used[r['id']]
        indices = r['pools'][config['pool']]
        assert len(indices) == len(set(indices))
        docs = [{k: r['documents'][i][k] for k in ['corpus_id', 'corpus_row', 'title', 'text']} for i in indices]
        assert len({d['corpus_id'] for d in docs}) == len(docs)
        rows.append({'id': r['id'], 'question': r['question'], 'documents': docs, 'source_document_indices': indices})
    files = ['configs/cqc_budget_stage12.json', 'configs/cqc_budget_stage12_prompts.json', 'research/stage12_cqc_budget_protocol.md', 'src/prepare_cqc_budget12.py', 'src/cqc_budget12_core.py', 'src/run_cqc_budget12.py', 'src/audit_cqc_budget12.py', 'src/test_cqc_budget12.py', 'src/qwen_student.py', 'configs/qwen512_v1.json', 'src/common.py', 'src/cqc_sampling12.py', 'src/test_cqc_sampling12.py', 'src/utility_student.py', 'src/verify_cqc_budget12_gpu.py', 'src/queue_cqc_budget12.py', 'src/run_idle_serial.py', 'src/launch_cqc_budget12.sh', 'src/test_cqc_budget12_audit.py', 'references/hotpot_eval.txt', 'references/cqc_rag_stage09/manifest.json', 'references/cqc_rag_stage09/inference/main_reranker_record_trivaiqa.py', 'references/cqc_rag_stage09/evaluate/autoregression_evalutor.py', 'results/reader_replay_stage10/inputs.jsonl', 'logs/qwen3_reader_stage10_assets_verified.json', 'logs/second_reader_assets_verified.json', 'configs/cqc_budget_stage12_reranker_assets.json']
    hashes = {name: file_sha(ROOT / name) for name in files}
    destination = base
    base = destination.with_name(destination.name + '.preparing')
    assert not base.exists()
    base.mkdir()
    write_new(base / 'inputs.jsonl', rows)
    manifest = {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_path': config['candidate_source'], 'source_sha256': file_sha(source), 'inputs_sha256': file_sha(base / 'inputs.jsonl'), 'files_sha256': hashes, 'ids': [r['id'] for r in rows], 'pool_counts': [len(r['documents']) for r in rows], 'questions': len(rows), 'generation_started_at_freeze': False, 'labels_in_inputs': False, 'scope': config['scope']}
    (base / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    os.rename(base, destination)
    print(json.dumps({'questions': len(rows), 'documents': sum(manifest['pool_counts']), 'frozen': True, 'labels_in_inputs': False}))


if __name__ == '__main__':
    main()
