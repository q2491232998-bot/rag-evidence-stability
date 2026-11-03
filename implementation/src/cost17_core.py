import datetime
from fractions import Fraction
import json
import os

from common import ROOT, atomic_json, digest, read_jsonl
from cqc_budget12_core import file_sha, source_path, write_new


def configuration():
    return json.loads((ROOT / 'configs/cost_stage17.json').read_text())


def select_indices(row, ranks, arm, cfg):
    n = len(row['documents'])
    if arm == 'full':
        return list(range(n))
    assert arm == 'rrf64'
    values = [sum((Fraction(1, cfg['rrf_constant'] + ranks[name][i]) for name in ['bge', 'e5'] if ranks[name][i] is not None), Fraction(0, 1)) for i in range(n)]
    return sorted(range(n), key=lambda i: (-values[i], i))[:cfg['shortlist_size']]


def make_schedule(rows, cfg):
    result = []
    original_indices = {r['id']: i for i, r in enumerate(rows)}
    for repeat in range(cfg['repetitions']):
        ordered = sorted(rows, key=lambda r: digest({'seed': cfg['seed'], 'repeat': repeat, 'id': r['id']}))
        for position, row in enumerate(ordered):
            arms = cfg['arms'] if (original_indices[row['id']] + repeat) % 2 == 0 else cfg['arms'][::-1]
            for slot, arm in enumerate(arms):
                result.append({'repeat': repeat, 'position': position, 'slot': slot, 'id': row['id'], 'arm': arm})
    return result


def build_ranks(rows, cfg):
    sources = {name: {r['id']: r for r in read_jsonl(source_path(cfg[name + '_source']))} for name in ['bge', 'e5']}
    ranks = []
    for row in rows:
        result = {'id': row['id']}
        union = set()
        for name in ['bge', 'e5']:
            original = sources[name][row['id']]
            assert original['question'] == row['question']
            documents = original['documents'][:cfg['retriever_depth']]
            mapping = {d['corpus_id']: (i + 1, d) for i, d in enumerate(documents)}
            assert len(mapping) == len(documents) == cfg['retriever_depth']
            union.update(mapping)
            result[name] = []
            for d in row['documents']:
                found = mapping.get(d['corpus_id'])
                result[name].append(found[0] if found else None)
                if found:
                    assert all(d[k] == found[1][k] for k in ['corpus_id', 'corpus_row', 'title', 'text'])
        assert union == {d['corpus_id'] for d in row['documents']}
        ranks.append(result)
    return ranks


def check17():
    cfg = configuration()
    base = ROOT / cfg['output']
    manifest = json.loads((base / 'manifest.json').read_text())
    for name, sha in manifest['files_sha256'].items():
        assert file_sha(source_path(name)) == sha, name
    for name, sha in manifest['inputs_sha256'].items():
        assert file_sha(base / name) == sha, name
    return cfg, base, manifest


def prepare():
    cfg = configuration()
    base = ROOT / cfg['output']
    staging = base.with_name(base.name + '.preparing')
    assert not base.exists() and not staging.exists()
    rows = read_jsonl(ROOT / cfg['candidate_source'])
    assert len(rows) == cfg['questions'] == 32
    assert sum(len(r['documents']) for r in rows) == 5631
    ranks = build_ranks(rows, cfg)
    files = ['configs/cost_stage17.json', 'research/stage17_cost_protocol.md', 'src/cost17_core.py', 'src/run_cost17.py', 'src/audit_cost17.py', 'src/test_cost17.py', 'src/queue_cost17.py', 'src/common.py', 'src/cqc_budget12_core.py', 'src/run_cqc_budget12.py', 'src/audit_cqc_budget12.py', 'src/qwen_student.py', 'src/utility_student.py', 'src/cqc_sampling12.py', 'src/run_idle_serial.py', 'configs/qwen512_v1.json', 'configs/cqc_budget_stage12_reranker_assets.json', 'logs/qwen3_reader_stage10_assets_verified.json', cfg['candidate_source'], cfg['bge_source'], cfg['e5_source'], 'references/hotpot_eval.txt', 'data/stage08_confirmation/labels.jsonl']
    files += ['results/cqc_budget_stage12_pilot32/' + n for n in ['manifest.json', 'rankings.jsonl', 'answers.jsonl', 'answer_cache.jsonl']]
    staging.mkdir()
    write_new(staging / 'inputs.jsonl', rows)
    write_new(staging / 'retrieval_ranks.jsonl', ranks)
    write_new(staging / 'schedule.jsonl', make_schedule(rows, cfg))
    record = {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'files_sha256': {n: file_sha(source_path(n)) for n in files}, 'inputs_sha256': {n: file_sha(staging / n) for n in ['inputs.jsonl', 'retrieval_ranks.jsonl', 'schedule.jsonl']}, 'generation_started': False, 'answer_labels_parsed': False, 'labels_hashed_as_opaque_bytes_only': True, 'prior_full_outcomes_known': True, 'questions': 32, 'documents': 5631, 'primary_reranker_calls': cfg['repetitions'] * (5631 + 32 * cfg['shortlist_size']), 'primary_generation_calls': 32 * 2 * cfg['repetitions']}
    atomic_json(staging / 'manifest.json', record)
    os.rename(staging, base)
    print(json.dumps({'state': 'frozen', 'manifest_sha256': file_sha(base / 'manifest.json'), 'counts': record['primary_reranker_calls']}))


if __name__ == '__main__':
    prepare()
