import collections
import hashlib
import json
import os
import pathlib
import re
import shutil
import string
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_jsonl(path):
    with pathlib.Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def atomic_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = ROOT / '.backups' / f'{path.name}.{time.time_ns()}'
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def normalize(text):
    text = text.lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return ' '.join(text.split())


def answer_metrics(prediction, answer):
    p, g = normalize(prediction), normalize(answer)
    em = float(p == g)
    if (p in ('yes', 'no', 'noanswer') or g in ('yes','no','noanswer')) and p != g:
        return {'em':em,'f1':0.0}
    overlap = sum((collections.Counter(p.split()) & collections.Counter(g.split())).values())
    if not overlap:
        return {'em':em,'f1':0.0}
    precision = overlap / len(p.split())
    recall = overlap / len(g.split())
    return {'em':em,'f1':2 * precision * recall / (precision + recall)}


def passage_text(passage):
    return f"Title: {passage['title']}\n{passage['text']}"


def context_text(passages, indices):
    return '\n\n'.join(passage_text(passages[i]) for i in indices)


def token_count(tokenizer, text):
    return len(tokenizer.encode(text,add_special_tokens=False))


def pack(passages, order, budget, tokenizer):
    selected = []
    for i in order:
        if token_count(tokenizer,context_text(passages,selected + [int(i)])) <= budget:
            selected.append(int(i))
    text = context_text(passages, selected)
    assert token_count(tokenizer,text) <= budget
    return selected, text


def make_prompt(tokenizer, question, context):
    text = f"Answer the question using the provided passages. Output only the short answer, with no explanation. If the answer is yes or no, output only yes or no.\n\nPassages:\n{context}\n\nQuestion: {question}"
    return tokenizer.apply_chat_template([{'role':'system','content':'You are a helpful question answering assistant.'},{'role':'user','content':text}],tokenize=False,add_generation_prompt=True)
