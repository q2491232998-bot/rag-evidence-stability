import argparse
import gc
import hashlib
import json
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from common import ROOT, atomic_json, digest, read_jsonl
from retrieval_math import merge_topk


def checksum(path):
    value = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--top-k', type=int, default=100)
    p.add_argument('--query-batch-size', type=int, default=64)
    p.add_argument('--allow-incomplete-index', action='store_true')
    args = p.parse_args()
    assert args.top_k > 0 and args.query_batch_size > 0
    folder = ROOT / 'indices/beir_hotpot_bge_small_v1'
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['complete'] or args.allow_incomplete_index, 'Incomplete index requires explicit smoke flag'
    assert manifest['shards']
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'retrieved.jsonl').exists():
        raise FileExistsError('Completed retrieval output already exists')
    rows = [{'id': r['id'], 'question': r['question']} for r in read_jsonl(ROOT / args.input)]
    assert len({r['id'] for r in rows}) == len(rows)
    torch.set_num_threads(4)
    torch.manual_seed(20261011)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = SentenceTransformer(str(ROOT / 'models/BAAI--bge-small-en-v1.5'), device='cuda', local_files_only=True)
    model.max_seq_length = manifest['config']['max_length']
    vectors = model.encode([manifest['config']['query_instruction'] + r['question'] for r in rows], batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-4)
    np.save(out / 'query_vectors.npy', vectors)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    query_vectors = torch.from_numpy(vectors).to('cuda')
    best_scores = [np.array([], dtype=np.float32) for _ in rows]
    best_indices = [np.array([], dtype=np.int64) for _ in rows]
    timing = []
    expected_start = 0
    for shard in manifest['shards']:
        shard_start = time.perf_counter()
        assert shard['first_global_index'] == expected_start
        path = folder / shard['vectors']
        assert checksum(path) == shard['vector_sha256']
        embeddings = np.load(path)
        assert embeddings.dtype == np.float32 and embeddings.shape == (shard['documents'], 384)
        assert np.isfinite(embeddings).all()
        corpus_vectors = torch.from_numpy(embeddings).to('cuda')
        with torch.inference_mode():
            for begin in range(0, len(rows), args.query_batch_size):
                scores = (query_vectors[begin:begin + args.query_batch_size] @ corpus_vectors.T).cpu().numpy()
                for offset, score in enumerate(scores):
                    q = begin + offset
                    best_scores[q], best_indices[q] = merge_topk(best_scores[q], best_indices[q], score, shard['first_global_index'], args.top_k)
        expected_start += shard['documents']
        timing.append({'shard': shard['index'], 'seconds': time.perf_counter() - shard_start})
        if shard['index'] % 20 == 0:
            print(json.dumps({'completed_shard': shard['index'], 'searched_documents': expected_start, 'seconds': time.perf_counter() - started}), flush=True)
        del corpus_vectors
    assert expected_start == manifest['documents']
    search_seconds = time.perf_counter() - started
    requested = {int(i) for local in best_indices for i in local}
    references = {}
    for shard in manifest['shards']:
        needed = requested & set(range(shard['first_global_index'], shard['first_global_index'] + shard['documents']))
        if not needed:
            continue
        path = folder / shard['references']
        assert checksum(path) == shard['reference_sha256']
        with path.open(encoding='utf-8') as f:
            for position, line in enumerate(f):
                global_index = shard['first_global_index'] + position
                if global_index in needed:
                    references[global_index] = json.loads(line)
    assert set(references) == requested
    passages = {}
    source = ROOT / 'data/raw/beir_hotpotqa/corpus.jsonl'
    with source.open('rb') as f:
        for index, reference in sorted(references.items(), key=lambda x: x[1]['byte_offset']):
            f.seek(reference['byte_offset'])
            row = json.loads(f.read(reference['byte_length']))
            assert row['_id'] == reference['id']
            passages[index] = {'corpus_id': row['_id'], 'title': row['title'], 'text': row['text'], 'corpus_row': index}
    with (out / 'retrieved.jsonl').open('w', encoding='utf-8') as f:
        for row, scores, indices in zip(rows, best_scores, best_indices):
            selected = [dict(passages[int(index)], score=float(score), rank=rank + 1) for rank, (score, index) in enumerate(zip(scores, indices))]
            f.write(json.dumps(dict(row, documents=selected), ensure_ascii=False) + '\n')
    atomic_json(out / 'runtime.json', {'queries': len(rows), 'documents_searched': manifest['documents'], 'index_complete': manifest['complete'], 'is_smoke_only': not manifest['complete'], 'top_k': args.top_k, 'input_hash_without_labels': digest(rows), 'index_config_hash': manifest['config_hash'], 'index_manifest_sha256': checksum(folder / 'manifest.json'), 'search_including_query_embedding_and_verification_seconds': search_seconds, 'total_seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'shard_timings': timing, 'label_access': False, 'tie_policy': 'Descending float32 cosine score, ascending corpus row index'})
    print(json.dumps({'queries': len(rows), 'documents_searched': manifest['documents'], 'index_complete': manifest['complete'], 'seconds': time.perf_counter() - started}), flush=True)


if __name__ == '__main__':
    main()
