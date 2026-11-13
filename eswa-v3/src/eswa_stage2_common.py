"""Shared paths and immutable I/O; deliberately no CUDA visibility changes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import datetime
from common import ROOT,read_jsonl,digest

CFG=json.loads((ROOT/'configs/eswa_gate_stage2_v1.json').read_text())
BASE=ROOT/CFG['output'];DATA=ROOT/CFG['data']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def check():
    assert os.name!='nt','Remote computation only'
    assert not (ROOT/'logs/research.pause').exists(),'Pause marker'
    assert shutil.disk_usage(ROOT).free>=CFG['minimum_free_bytes'],'Disk headroom'
    if BASE.exists():
        assert sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())<CFG['maximum_new_result_bytes'],'Result size cap'


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)


def save_rows(path,rows):
    with Path(path).open('x',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def key(text):return hashlib.sha256((str(CFG['seed'])+'|'+text).encode()).hexdigest()
