import argparse
import itertools
import json
import math
import random
import time

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import ROOT, atomic_json, context_text, digest, read_jsonl
from utility_student import grouped_utilities, inference_example, subset_list


class Scorer:
    def __init__(self, config, adapter=None, train=False):
        self.config = config
        path = str(ROOT / 'models' / config['base_model'].replace('/', '--'))
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left', local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation='sdpa').to('cuda')
        if train:
            self.model = get_peft_model(self.model, LoraConfig(task_type='CAUSAL_LM', r=config['lora_rank'], lora_alpha=config['lora_alpha'], target_modules=config['lora_targets'], lora_dropout=0., bias='none'))
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            self.model.train()
        elif adapter:
            self.model = PeftModel.from_pretrained(self.model, str(adapter))
            self.model.eval()
        else:
            self.model.eval()
        self.no = self.tokenizer.convert_tokens_to_ids('no')
        self.yes = self.tokenizer.convert_tokens_to_ids('yes')
        self.prefix = self.tokenizer.encode('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n', add_special_tokens=False)
        self.suffix = self.tokenizer.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', add_special_tokens=False)
        self.truncated = 0

    def encode_one(self, row, subset):
        context = context_text(row['passages'], subset) or 'No passages.'
        text = f"<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: {row['question']}\n<Document>: {context}"
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        limit = self.config['max_length'] - len(self.prefix) - len(self.suffix)
        self.truncated += int(len(tokens) > limit)
        return self.prefix + tokens[:limit] + self.suffix

    def score(self, encoded):
        batch = self.tokenizer.pad({'input_ids': encoded}, padding=True, return_tensors='pt').to('cuda')
        logits = self.model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
        return logits[:, self.yes] - logits[:, self.no]


def manifest(config, out, frozen=False):
    split = json.loads((ROOT / config['split_manifest']).read_text())
    value = {'config': config, 'config_hash': digest(config), 'fit_ids': split['fit_ids'], 'calibration_ids': split['calibration_ids'], 'encoder_training': not frozen, 'fit_ids_role': 'comparison partition only, not encoder training' if frozen else 'encoder training', 'prompt_source': 'https://huggingface.co/Qwen/Qwen3-Reranker-0.6B', 'target_policy': 'Within-query teacher utility differences; both pair orders share averaged pair utility; labels never used in inference.'}
    atomic_json(out / 'training_manifest.json', value)
    return value


def train(config, out):
    if (out / 'checkpoint').exists() or (out / 'training_manifest.json').exists():
        raise FileExistsError('Training output already exists')
    split = manifest(config, out)
    rows = {r['id']: inference_example(r) for r in read_jsonl(ROOT / config['training_input'])}
    ranks = {r['id']: r for r in read_jsonl(ROOT / config['training_rankings'])}
    labels = grouped_utilities(ROOT / config['training_utilities'])
    scorer = Scorer(config, train=True)
    cache = {}
    pairs = []
    label_sets = {}
    for q in split['fit_ids']:
        subsets = subset_list(ranks[q]['orders']['dense'][:config['n_candidates']])
        cache[q] = [scorer.encode_one(rows[q], s) for s in subsets]
        targets = [(labels[q][s] + labels[q][s[::-1]]) / 2 if len(s) == 2 else labels[q][s] for s in subsets]
        label_sets[q] = (subsets, targets)
        comparisons = [(i, j) for i, j in itertools.combinations(range(len(subsets)), 2) if abs(targets[i] - targets[j]) > config['ranking_min_gap']]
        assert comparisons, q
        for epoch in range(config['epochs']):
            rng = random.Random(digest({'seed': config['seed'], 'q': q, 'epoch': epoch}))
            chosen = rng.sample(comparisons, min(len(comparisons), config['preference_pairs_per_query']))
            pairs.extend((epoch, q, i, j, targets[i] - targets[j]) for i, j in chosen)
    trainable = [p for p in scorer.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    epoch_tasks = [[r for r in pairs if r[0] == epoch] for epoch in range(config['epochs'])]
    batches_per_epoch = [math.ceil(len(r) / config['preference_batch_size']) for r in epoch_tasks]
    total_updates = sum(math.ceil(n / config['accumulate_batches']) for n in batches_per_epoch)
    warmup = max(1, round(total_updates * .1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / warmup, max(0., (total_updates - step) / max(1, total_updates - warmup))))
    atomic_json(out / 'preference_manifest.json', {'pairs': pairs, 'pairs_sha256': digest(pairs), 'trainable_parameters': sum(p.numel() for p in trainable), 'unique_encoded_inputs': sum(len(r) for r in cache.values()), 'truncated_unique_inputs': scorer.truncated, 'optimizer_updates': total_updates})
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    history = []
    optimizer.zero_grad(set_to_none=True)
    for epoch, tasks in enumerate(epoch_tasks):
        random.Random(config['seed'] + epoch).shuffle(tasks)
        losses = []
        for batch_index, begin in enumerate(range(0, len(tasks), config['preference_batch_size'])):
            batch = tasks[begin:begin + config['preference_batch_size']]
            encoded = [cache[q][i] for _, q, i, j, target in batch] + [cache[q][j] for _, q, i, j, target in batch]
            target = torch.tensor([r[-1] for r in batch], device='cuda') / config['utility_scale']
            prediction = scorer.score(encoded)
            difference = (prediction[:len(batch)] - prediction[len(batch):]) / config['utility_scale']
            regression = torch.nn.functional.smooth_l1_loss(difference, target)
            ranking = torch.nn.functional.softplus(-difference * torch.sign(target)).mean()
            loss = regression + config['ranking_weight'] * ranking
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            group_begin = batch_index // config['accumulate_batches'] * config['accumulate_batches']
            denominator = min(config['accumulate_batches'], batches_per_epoch[epoch] - group_begin)
            (loss / denominator).backward()
            losses.append(float(loss.detach()))
            if (batch_index + 1) % config['accumulate_batches'] == 0 or batch_index + 1 == batches_per_epoch[epoch]:
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
                assert torch.isfinite(norm) and norm > 0
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if batch_index % 100 == 0:
                print(json.dumps({'epoch': epoch + 1, 'batch': batch_index + 1, 'batches': batches_per_epoch[epoch], 'loss': float(np.mean(losses)), 'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}), flush=True)
        history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses))})
    scorer.model.save_pretrained(out / 'checkpoint', safe_serialization=True)
    scorer.tokenizer.save_pretrained(out / 'checkpoint')
    atomic_json(out / 'training_runtime.json', {'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'history': history, 'gpu': torch.cuda.get_device_name(0), 'selection_uses_development': False})


def predict(config, out, frozen):
    if frozen:
        split = manifest(config, out, frozen=True)
    else:
        split = json.loads((out / 'training_manifest.json').read_text())
    scorer = Scorer(config, adapter=None if frozen else out / 'checkpoint')
    for name, input_path, rank_path in [('calibration', config['training_input'], config['training_rankings']), ('development', 'data/hotpot_development64.jsonl', 'results/development64_v1/rankings.jsonl')]:
        destination = out / (name + '_predictions.jsonl')
        if destination.exists():
            raise FileExistsError(destination)
        rows = [inference_example(r) for r in read_jsonl(ROOT / input_path)]
        if name == 'calibration':
            rows = [r for r in rows if r['id'] in split['calibration_ids']]
            assert len(rows) == len(split['calibration_ids'])
        else:
            assert not {r['id'] for r in rows} & set(split['fit_ids'] + split['calibration_ids'])
        ranks = {r['id']: r for r in read_jsonl(ROOT / rank_path)}
        tasks = [(r, s, scorer.encode_one(r, s)) for r in rows for s in subset_list(ranks[r['id']]['orders']['dense'][:config['n_candidates']])]
        tasks.sort(key=lambda r: len(r[2]))
        scores = {r['id']: [] for r in rows}
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            for begin in range(0, len(tasks), config['inference_batch_size']):
                batch = tasks[begin:begin + config['inference_batch_size']]
                values = scorer.score([r[2] for r in batch]).cpu().tolist()
                for (row, subset, _), value in zip(batch, values):
                    scores[row['id']].append({'indices': list(subset), 'utility': value})
        with destination.open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps({'id': row['id'], 'candidate_indices': ranks[row['id']]['orders']['dense'][:config['n_candidates']], 'input_hash': digest(row), 'config_hash': digest(config), 'scores': scores[row['id']]}) + '\n')
        atomic_json(destination.with_suffix('.runtime.json'), {'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'queries': len(rows), 'scored_inputs': len(tasks), 'truncated_inputs_cumulative': scorer.truncated, 'label_access': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['train', 'predict'], required=True)
    p.add_argument('--frozen', action='store_true')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    config = json.loads((ROOT / 'configs/qwen512_v1.json').read_text())
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.set_num_threads(4)
    train(config, out) if args.stage == 'train' else predict(config, out, args.frozen)
