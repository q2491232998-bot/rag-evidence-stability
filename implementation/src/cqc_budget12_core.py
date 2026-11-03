import collections
import hashlib
import json
import math
import os
from pathlib import Path
import re

from common import ROOT, digest, normalize


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def settings():
    return json.loads((ROOT / 'configs/cqc_budget_stage12.json').read_text())


def prompts():
    return json.loads((ROOT / 'configs/cqc_budget_stage12_prompts.json').read_text())


def source_path(name):
    for path in [ROOT / name, ROOT / 'snapshots/stage09_e5' / name]:
        if path.exists():
            return path
    raise FileNotFoundError(name)


def write_new(path, rows):
    path = Path(path)
    assert not path.exists(), str(path)
    temp = path.with_name(path.name + '.pending')
    with temp.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temp, path)


def parse_paraphrases(raw, original, n=4):
    found = []
    for line in raw.splitlines():
        match = re.match(r'^\s*\d+[.)]\s*(.+)', line)
        if match:
            value = match.group(1).strip()
            if not value.endswith('?'):
                value += '?'
            found.append(value)
    used = found[:n]
    fallback = n - len(used)
    return used + [original] * fallback, {'parsed_count': len(found), 'fallback_count': fallback}


def parse_answer(raw):
    text = raw.strip()
    match = re.search(r'Evidence:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    head = text[:match.start()].strip() if match else text
    answer = re.sub(r'^Answer:\s*', '', head, flags=re.IGNORECASE).strip()
    return {'answer': answer, 'evidence': match.group(1).strip() if match else '', 'format_has_evidence': match is not None}


def paragraph(document):
    return 'Title: ' + document['title'] + '\n' + document['text']


def pack_topk(documents, order, tokenizer, budget, k):
    chosen = []
    for i in order[:k]:
        trial = chosen + [i]
        text = '\n\n'.join(paragraph(documents[j]) for j in trial)
        if len(tokenizer.encode(text, add_special_tokens=False)) <= budget:
            chosen = trial
    text = '\n\n'.join(paragraph(documents[j]) for j in chosen)
    return {'selected': chosen, 'context': text, 'context_tokens': len(tokenizer.encode(text, add_special_tokens=False))}


def answer_messages(question, context, neutral=False):
    if neutral:
        user = 'Answer the question using the provided passages. Output only the short answer, with no explanation.\n\nPassages:\n' + context + '\n\nQuestion: ' + question
        return [{'role': 'system', 'content': 'You are a helpful question answering assistant.'}, {'role': 'user', 'content': user}]
    return [{'role': 'system', 'content': prompts()['answer_system']}, {'role': 'user', 'content': 'Documents:\n' + context + '\n\nQuestion: ' + question}]


def verification_prompt(question, answer, evidence):
    return prompts()['verification_template'].format(question=question, answer=answer, evidence=evidence)


def qwen_prompt(tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    assert ids == tokenizer(text, add_special_tokens=False).input_ids
    return text, ids


def binary_ids(tokenizer):
    result = {}
    for word in ['Yes', 'No']:
        plain = tokenizer.encode(word, add_special_tokens=False)
        spaced = tokenizer.encode(' ' + word, add_special_tokens=False)
        assert len(spaced) == 1 or len(plain) == 1
        result[word] = {'plain': plain, 'spaced': spaced, 'chosen': spaced[0] if len(spaced) == 1 else plain[0]}
    assert result['Yes']['chosen'] != result['No']['chosen']
    return result


def sigmoid(value):
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def select_rules(candidates, probability, lambda0=.5):
    n = len(candidates)
    assert n == len(probability) == 5 and all(len(r) == n for r in probability)
    counts = collections.Counter(normalize(r['answer']) for r in candidates)
    values = {k: [] for k in ['vote', 'own_verifier', 'original_query_verifier', 'cross_mean', 'cqc_code_std', 'cqc_paper_var']}
    for i, row in enumerate(candidates):
        p = probability[i]
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in p)
        mean = sum(p) / n
        variance = sum((v - mean) ** 2 for v in p) / n
        score = {'vote': counts[normalize(row['answer'])], 'own_verifier': p[i], 'original_query_verifier': p[0], 'cross_mean': mean, 'cqc_code_std': max(0., min(1., mean - (.5 if mean > .7 else .2) * math.sqrt(variance))), 'cqc_paper_var': mean - lambda0 * mean * variance}
        for key in values:
            values[key].append(score[key] if row['answer'] else -1e9)
    choices = {key: max(range(n), key=lambda i: (v[i], -i)) for key, v in values.items()}
    return choices, values


def check_freeze(base):
    manifest = json.loads((base / 'manifest.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        assert file_sha(ROOT / name) == expected, name
    assert file_sha(base / 'inputs.jsonl') == manifest['inputs_sha256']
    return manifest
