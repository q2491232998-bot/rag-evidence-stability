import argparse
import gc
import itertools
import json
import math
import random
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import ROOT, atomic_json, context_text, digest, read_jsonl


def inference_example(row):
    return {'id': row['id'], 'question': row['question'], 'passages': [{'title': p['title'], 'text': p['text']} for p in row['passages']]}


def subset_list(candidates):
    return [()] + [(i,) for i in candidates] + [s for pair in itertools.combinations(candidates, 2) for s in [pair, pair[::-1]]]


def grouped_utilities(path):
    grouped = {}
    for row in read_jsonl(path):
        value = float(row['utility'])
        if not np.isfinite(value):
            raise ValueError('Nonfinite utility label')
        key = tuple(row['indices'])
        local = grouped.setdefault(row['id'], {})
        if key in local:
            raise ValueError('Duplicate utility key')
        local[key] = value
    return grouped


def encode(tokenizer, row, subsets, max_length, device=None):
    result = tokenizer([row['question']] * len(subsets), [context_text(row['passages'], s) or 'No passages.' for s in subsets], padding=True, truncation='only_second', max_length=max_length, return_tensors='pt')
    return result.to(device) if device else result


def train(config, out):
    if (out / 'checkpoint').exists():
        raise FileExistsError('A student checkpoint already exists')
    rows = {r['id']: inference_example(r) for r in read_jsonl(ROOT / config['training_input'])}
    labels = grouped_utilities(ROOT / config['training_utilities'])
    ranks = {r['id']: r for r in read_jsonl(ROOT / config['training_rankings'])}
    ids = sorted(rows, key=lambda q: digest({'seed': config['seed'], 'split': 'student_calibration', 'id': q}))
    calibration_ids = ids[-config['calibration_queries']:]
    fit_ids = ids[:-config['calibration_queries']]
    assert not set(fit_ids) & set(calibration_ids)
    model_path = str(ROOT / 'models' / config['base_model'].replace('/', '--'))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to('cuda')
    batches = {}
    truncated = 0
    examples = 0
    for q in fit_ids:
        subsets = subset_list(ranks[q]['orders']['dense'][:config['n_candidates']])
        target = np.array([(labels[q][s] + labels[q][s[::-1]]) / 2 if len(s) == 2 else labels[q][s] for s in subsets], dtype=np.float32)
        encoded = encode(tokenizer, rows[q], subsets, config['max_length'])
        batches[q] = (encoded, torch.tensor(target))
        lengths = tokenizer([rows[q]['question']] * len(subsets), [context_text(rows[q]['passages'], s) or 'No passages.' for s in subsets], truncation=False, return_length=True)['length']
        truncated += sum(n > config['max_length'] for n in lengths)
        examples += len(subsets)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    updates_per_epoch = math.ceil(len(fit_ids) / config['accumulate_queries'])
    total_updates = config['epochs'] * updates_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / max(1, round(total_updates * .1)), max(0, (total_updates - step) / max(1, total_updates - round(total_updates * .1)))))
    metadata = {'config': config, 'config_hash': digest(config), 'fit_ids': fit_ids, 'calibration_ids': calibration_ids, 'fit_examples_per_epoch': examples, 'truncated_examples_per_epoch': truncated, 'label_policy': 'mean-centered within query; averaged pair orders; empty and singleton and pair only; no high-order input in encoder training'}
    atomic_json(out / 'training_manifest.json', metadata)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    history = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(config['epochs']):
        order = list(fit_ids)
        random.Random(config['seed'] + epoch).shuffle(order)
        model.train()
        losses = []
        for step, q in enumerate(order):
            inputs, target = batches[q]
            target = target.to('cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = model(**inputs.to('cuda')).logits.reshape(-1).float()
                centered = (pred - pred.mean()) / config['utility_scale']
                truth = (target - target.mean()) / config['utility_scale']
                regression = torch.nn.functional.smooth_l1_loss(centered, truth)
                differences = target[:, None] - target[None, :]
                mask = differences > config['ranking_min_gap']
                ranking = torch.nn.functional.softplus(-(pred[:, None] - pred[None, :])[mask] / config['utility_scale']).mean() if mask.any() else pred.sum() * 0
                loss = regression + config['ranking_weight'] * ranking
            group_start = step // config['accumulate_queries'] * config['accumulate_queries']
            denominator = min(config['accumulate_queries'], len(order) - group_start)
            (loss / denominator).backward()
            losses.append(float(loss.detach()))
            if (step + 1) % config['accumulate_queries'] == 0 or step + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            inputs.to('cpu')
            if step % 40 == 0:
                print(json.dumps({'epoch': epoch + 1, 'query': step + 1, 'total': len(order), 'mean_loss': float(np.mean(losses)), 'seconds': time.perf_counter() - start}), flush=True)
        history.append({'epoch': epoch + 1, 'mean_loss': float(np.mean(losses))})
    model.save_pretrained(out / 'checkpoint', safe_serialization=True)
    tokenizer.save_pretrained(out / 'checkpoint')
    atomic_json(out / 'training_runtime.json', {'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'gpu': torch.cuda.get_device_name(0), 'history': history})


def predict(config, out, input_path, rank_path, output_path, calibration_only=False):
    if output_path.exists():
        raise FileExistsError(output_path)
    rows = [inference_example(r) for r in read_jsonl(input_path)]
    if calibration_only:
        ids = set(json.loads((out / 'training_manifest.json').read_text())['calibration_ids'])
        rows = [r for r in rows if r['id'] in ids]
        assert len(rows) == len(ids)
    ranks = {r['id']: r for r in read_jsonl(rank_path)}
    tokenizer = AutoTokenizer.from_pretrained(str(out / 'checkpoint'), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(out / 'checkpoint'), local_files_only=True).to('cuda').eval()
    items = [(r, s) for r in rows for s in subset_list(ranks[r['id']]['orders']['dense'][:config['n_candidates']])]
    items.sort(key=lambda x: len(context_text(x[0]['passages'], x[1])))
    scores = {r['id']: {} for r in rows}
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    truncated = 0
    for start in range(0, len(items), config['inference_batch_size']):
        batch = items[start:start + config['inference_batch_size']]
        questions = [r['question'] for r, s in batch]
        contexts = [context_text(r['passages'], s) or 'No passages.' for r, s in batch]
        lengths = tokenizer(questions, contexts, truncation=False, return_length=True)['length']
        truncated += sum(n > config['max_length'] for n in lengths)
        encoded = tokenizer(questions, contexts, padding=True, truncation='only_second', max_length=config['max_length'], return_tensors='pt').to('cuda')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(**encoded).logits.reshape(-1).float().cpu().tolist()
        for (r, subset), value in zip(batch, pred):
            scores[r['id']][tuple(subset)] = value
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - begin
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        for row in rows:
            local = scores[row['id']]
            candidates = ranks[row['id']]['orders']['dense'][:config['n_candidates']]
            record = {'id': row['id'], 'candidate_indices': candidates, 'input_hash': digest(row), 'config_hash': digest(config), 'scores': [{'indices': list(s), 'utility': v} for s, v in local.items()]}
            f.write(json.dumps(record) + '\n')
    atomic_json(output_path.with_suffix('.runtime.json'), {'seconds': elapsed, 'queries': len(rows), 'scored_inputs': len(items), 'truncated_inputs': truncated, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'inference_input_fields': ['question', 'passages.title', 'passages.text'], 'label_access': False, 'checkpoint_config_hash': digest(config)})


def set_features(values, subset):
    if not subset:
        return np.zeros(6), np.zeros(13)
    empty = values[()]
    singles = np.array([values[(i,)] - empty for i in subset])
    pairs = np.array([(values[(i, j)] + values[(j, i)]) / 2 - values[(i,)] - values[(j,)] + empty for i, j in itertools.combinations(subset, 2)])
    first = np.array([len(subset), singles.sum(), singles.mean(), singles.max(), singles.min(), singles.std()])
    second = [pairs.sum(), pairs.mean(), pairs.max(), pairs.min(), pairs.std(), np.maximum(pairs, 0).sum(), np.minimum(pairs, 0).sum()] if len(pairs) else [0.] * 7
    return first, np.concatenate([first, second])


def shuffle_pairs(values, candidates, q, seed):
    shuffled = dict(values)
    pairs = list(itertools.combinations(candidates, 2))
    interactions = [(values[(i, j)] + values[(j, i)]) / 2 - values[(i,)] - values[(j,)] + values[()] for i, j in pairs]
    rng = np.random.default_rng(int(digest({'seed': seed, 'query': q})[:16], 16))
    rng.shuffle(interactions)
    for (i, j), effect in zip(pairs, interactions):
        shuffled[(i, j)] = shuffled[(j, i)] = effect + values[(i,)] + values[(j,)] - values[()]
    return shuffled


def calibrated_predict(model, features):
    return ((np.asarray(features) - np.array(model['mean'])) / np.array(model['scale'])) @ np.array(model['coef']) + model['intercept']


def calibrate(config, out):
    path = out / 'calibrators.json'
    if path.exists():
        raise FileExistsError(path)
    predictions = read_jsonl(out / 'calibration_predictions.jsonl')
    labels = grouped_utilities(ROOT / config['training_utilities'])
    manifest = json.loads((out / 'training_manifest.json').read_text())
    assert set(r['id'] for r in predictions) == set(manifest['calibration_ids'])
    matrices = {m: [] for m in ['singleton', 'pair', 'shuffled_pair']}
    targets = []
    keys = []
    for row in predictions:
        values = {tuple(r['indices']): r['utility'] for r in row['scores']}
        shuffled = shuffle_pairs(values, row['candidate_indices'], row['id'], config['seed'])
        truth = labels[row['id']]
        for subset in sorted(truth):
            if len(subset) == 2 and subset[0] > subset[1]:
                continue
            a, b = set_features(values, subset)
            matrices['singleton'].append(a)
            matrices['pair'].append(b)
            matrices['shuffled_pair'].append(set_features(shuffled, subset)[1])
            value = (truth[subset] + truth[subset[::-1]]) / 2 if len(subset) == 2 else truth[subset]
            targets.append(value - truth[()])
            keys.append([row['id'], list(subset)])
    fitted = {}
    for name, x in matrices.items():
        scaler = StandardScaler().fit(x)
        model = Ridge(alpha=config['ridge_alpha']).fit(scaler.transform(x), targets)
        fitted[name] = {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'coef': model.coef_.tolist(), 'intercept': float(model.intercept_)}
    atomic_json(path, {'config_hash': digest(config), 'fit_ids': manifest['calibration_ids'], 'n_subsets': len(targets), 'keys_hash': digest(keys), 'models': fitted, 'gold_used_only_for_training_targets': True})


def evaluate(config, out, prediction_path, label_path, report_path):
    predictions = read_jsonl(prediction_path)
    labels = grouped_utilities(label_path)
    manifest = json.loads((out / 'training_manifest.json').read_text())
    assert not set(r['id'] for r in predictions) & set(manifest['fit_ids'] + manifest['calibration_ids'])
    models = json.loads((out / 'calibrators.json').read_text())['models']
    per_query = []
    for row in predictions:
        values = {tuple(r['indices']): r['utility'] for r in row['scores']}
        truth = labels[row['id']]
        shuffled = shuffle_pairs(values, row['candidate_indices'], row['id'], config['seed'])
        subsets = [s for s in truth if len(s) >= 3]
        if not subsets:
            continue
        target = np.array([truth[s] - truth[()] for s in subsets])
        features = [set_features(values, s) for s in subsets]
        local = {'id': row['id'], 'subsets': len(subsets)}
        for name, x in [('singleton', [f[0] for f in features]), ('pair', [f[1] for f in features]), ('shuffled_pair', [set_features(shuffled, s)[1] for s in subsets])]:
            pred = calibrated_predict(models[name], x)
            correlation = float(spearmanr(pred, target).statistic)
            local[name] = {'rmse': float(np.sqrt(np.mean((pred - target) ** 2))), 'spearman': correlation if np.isfinite(correlation) else None}
        per_query.append(local)
    summary = {m: {metric: float(np.mean([r[m][metric] for r in per_query if r[m][metric] is not None])) for metric in ['rmse', 'spearman']} for m in models}
    report = {'n_queries_eligible': len(per_query), 'metrics': summary, 'per_query': per_query, 'interpretation': 'Predicted features on disjoint development queries; teacher labels accessed for evaluation only, not inference.'}
    atomic_json(report_path, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'per_query'}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['train', 'predict', 'calibrate', 'evaluate'], required=True)
    p.add_argument('--config', default='configs/student512_v1.json')
    p.add_argument('--output', default='results/student512_v1')
    p.add_argument('--input')
    p.add_argument('--ranks')
    p.add_argument('--predictions')
    p.add_argument('--labels')
    p.add_argument('--report', default='results/student512_v1/development_utility.json')
    p.add_argument('--calibration-only', action='store_true')
    args = p.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.set_num_threads(4)
    if args.stage == 'train':
        train(config, out)
    elif args.stage == 'predict':
        predict(config, out, ROOT / args.input, ROOT / args.ranks, ROOT / args.predictions, args.calibration_only)
    elif args.stage == 'calibrate':
        calibrate(config, out)
    else:
        evaluate(config, out, ROOT / args.predictions, ROOT / args.labels, ROOT / args.report)


if __name__ == '__main__':
    main()
