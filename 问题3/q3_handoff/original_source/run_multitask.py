"""Independent Accuracy-first EMT runner. Formal work requires explicit train/evaluate."""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import json
import logging
import math
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parent
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from run_baseline import configure, make_optimizer, save_csv
from run_missing_standardized import _smoke_indices
from dataset_construction.common import sha256_file
from robust.data import MixDataStore
from standardized.normalization import fit_preprocessing, validate_preprocessing
from multitask.config import default_config, validate_config
from multitask.model import build_model, forward_training, predict_batch
from multitask.losses import compute_losses
from multitask.selection import selection_protocol, selection_score, EarlyStopping
from multitask.evaluation import evaluate, save_records, metrics_from_records, validate_records
from multitask.provenance import CODE_VERSION, implementation_provenance
from multitask.checkpoint import save_checkpoint, load_checkpoint, load_for_prediction

LOGGER = logging.getLogger('multitask')
MIXED = 'main_mixed/mixed_v1_s0'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def path_in_project(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def require_new_output(output):
    if Path(output).exists():
        raise FileExistsError(f'Refusing to overwrite an existing run: {output}')


class CleanTrainDataset(Dataset):
    """Use the accepted reader's original train view; expose no auxiliary data."""
    def __init__(self, paired):
        self.paired = paired

    def __len__(self):
        return len(self.paired)

    def __getitem__(self, index):
        item = self.paired[index]
        return {**item['reference'], **{key: item[key] for key in ('labels', 'index', 'id')}}


def make_loader(dataset, config, train=False, ordered=False):
    return DataLoader(dataset, batch_size=config['batch_size'] if train else config['eval_batch_size'],
                      shuffle=train and not ordered, drop_last=False, num_workers=config['num_workers'],
                      pin_memory=config['device'] == 'cuda',
                      generator=torch.Generator().manual_seed(config['seed']))


def training_dataset(store, method):
    paired = store.dataset('train', training=True)
    return CleanTrainDataset(paired) if method == 'M0' else paired


def selection_datasets(store, method, split='valid'):
    result = {'clean': store.dataset(split, family='paired', condition_id='complete')}
    if method != 'M0':
        result[MIXED] = store.dataset(split)
    return result


def gradient_evidence(model, method):
    prefixes = ['text_model', 'audio_model', 'video_model', 'fusion', 'post_fusion_layer_3', 'cls_head']
    if method == 'M2':
        prefixes += ['recon_text', 'recon_audio', 'recon_video', 'gmc_tokens_predictor',
                     'text_predictor', 'audio_predictor', 'video_predictor']
    evidence = {}
    for prefix in prefixes:
        grads = [p.grad for name, p in model.named_parameters()
                 if name.startswith(prefix + '.') and p.grad is not None]
        if not grads or any(not bool(torch.isfinite(g).all()) for g in grads):
            raise RuntimeError('Missing/nonfinite gradients: ' + prefix)
        norm = math.sqrt(sum(float(g.detach().float().square().sum()) for g in grads))
        if norm <= 0:
            raise RuntimeError('Zero gradient: ' + prefix)
        evidence[prefix] = norm
    if method != 'M2' and any(p.grad is not None for name, p in model.named_parameters()
                             if 'projector' in name or 'predictor' in name or name.startswith('recon_')):
        raise RuntimeError('Inactive DLFR heads received gradients')
    return evidence


def verify_optimizer(model, optimizer):
    optimized = [id(p) for group in optimizer.param_groups for p in group['params']]
    expected = [id(p) for p in model.parameters() if p.requires_grad]
    if len(set(optimized)) != len(optimized) or set(optimized) != set(expected):
        raise RuntimeError('Optimizer does not contain each trainable parameter exactly once')
    for parameter in model.cls_head.parameters():
        if not parameter.requires_grad or id(parameter) not in optimized:
            raise RuntimeError('Classification head is frozen or absent from optimizer')


def train_epoch(model, batches, optimizer, config, device, check_gradients=False):
    """Normalize each accumulation group by its actual samples, including tails."""
    if batches.drop_last:
        raise ValueError('Training must retain the tail batch')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    seen, steps, sums, evidence, sizes = 0, 0, {}, None, []
    accumulation = config['gradient_accumulation_steps']
    # Buffer only batch dictionaries, never multiple model graphs. This handles
    # arbitrary physical batch sizes without guessing from configured batch_size.
    iterator = iter(batches)
    while True:
        group = []
        for _ in range(accumulation):
            batch = next(iterator, None)
            if batch is None:
                break
            group.append(batch)
        if not group:
            break
        sample_count = sum(int(batch['labels'].numel()) for batch in group)
        for batch in group:
            outputs = forward_training(model, batch, device)
            losses = compute_losses(outputs, batch, config, device)
            if any(value.ndim != 0 or not bool(torch.isfinite(value)) for value in losses.values()):
                raise RuntimeError('Loss components must be finite scalars')
            n = int(batch['labels'].numel())
            (losses['total'] * (n / sample_count)).backward()
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach()) * n
            seen += n
            sizes.append(n)
            del outputs, losses
        if check_gradients:
            evidence = gradient_evidence(model, config['method'])
        torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
    if seen != len(batches.dataset) or steps < 1:
        raise RuntimeError('Training lost samples or failed to update')
    return {'n': seen, 'optimizer_steps': steps, **{k: v / seen for k, v in sums.items()}}, evidence, sizes


def evaluate_conditions(model, datasets, config, device, split='valid'):
    conditions, records = {}, {}
    for name, dataset in datasets.items():
        conditions[name], records[name] = evaluate(model, make_loader(dataset, config), device, split)
    identities = [[(row['index'], row['id'], row['y_true']) for row in rows] for rows in records.values()]
    if any(value != identities[0] for value in identities[1:]):
        raise ValueError('Selection conditions must use identical samples and labels')
    return conditions, records


def save_condition_results(output, split, conditions, records, *, overwrite=False):
    for name, metrics in conditions.items():
        stem = split + '__' + name.replace('/', '__')
        save_records(output / 'details' / (stem + '.csv'), records[name], overwrite=overwrite)
        write_json(output / 'details' / (stem + '.json'), metrics)


def setup_data(config):
    return MixDataStore(path_in_project(config['artifact_path']), path_in_project(config['data_path']))


def initialize_run(output, config, store, preprocessing, purpose):
    write_json(output / 'config.json', config)
    write_json(output / 'preprocessing.json', preprocessing)
    write_json(output / 'provenance.json', {'code_version': CODE_VERSION, 'purpose': purpose,
               'code_sha256': implementation_provenance(ROOT), 'data_binding': store.binding(),
               'selection_protocol': selection_protocol(config['method'])})


def train(config, output):
    config = validate_config(dict(config))
    output = Path(output)
    require_new_output(output)
    device = configure(config, output)
    store = setup_data(config)
    preprocessing = fit_preprocessing(store.source_path)
    validate_preprocessing(preprocessing, source_sha256=store.binding()['data_sha256'])
    initialize_run(output, config, store, preprocessing, 'train')
    dataset = training_dataset(store, config['method'])
    valid_datasets = selection_datasets(store, config['method'])
    # The formal selection cohort must not silently become a bounded probe.
    if len(dataset) != 3395 or any(len(value) != 728 for value in valid_datasets.values()):
        raise ValueError('Formal train/valid cohorts must contain 3395/728 samples')
    batches = make_loader(dataset, config, train=True)
    model = build_model(config, ROOT, preprocessing).to(device)
    optimizer = make_optimizer(model, config)
    verify_optimizer(model, optimizer)
    stopper = EarlyStopping(patience=config['early_stop_patience'])
    history, stop_reason = [], 'max_epochs'
    for epoch in range(1, config['epochs'] + 1):
        losses, _, _ = train_epoch(model, batches, optimizer, config, device)
        conditions, records = evaluate_conditions(model, valid_datasets, config, device)
        score = selection_score(config['method'], conditions)
        improved = stopper.update(score, epoch)
        history.append({'epoch': epoch, 'train': losses, 'valid': conditions,
                        'selection_score': score, 'improved': improved, 'best_epoch': stopper.best_epoch})
        if improved:
            save_checkpoint(output / 'best.pt', model, config, epoch, conditions, store.binding(), ROOT)
            save_condition_results(output, 'valid', conditions, records, overwrite=True)
        write_json(output / 'history.json', history)
        flattened = [{'epoch': row['epoch'], 'best_epoch': row['best_epoch'],
                      **{'train_' + k: v for k, v in row['train'].items()},
                      **{name.replace('/', '_') + '_' + key: value[key]
                         for name, value in row['valid'].items() for key in ('Accuracy', 'Macro_F1', 'MAE', 'Pearson')}}
                     for row in history]
        save_csv(output / 'details/history.csv', flattened)
        LOGGER.info('Epoch %d selection=%s best=%d', epoch, score, stopper.best_epoch)
        if stopper.should_stop:
            stop_reason = 'early_stopping'
            break
    del optimizer, model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    model = load_for_prediction(output / 'best.pt', ROOT, device=device)
    conditions, records = evaluate_conditions(model, valid_datasets, config, device)
    saved = read_json(output / 'history.json')[stopper.best_epoch - 1]['valid']
    for condition in conditions:
        for key in ('Accuracy', 'Macro_F1', 'MAE'):
            if not math.isclose(conditions[condition][key], saved[condition][key], abs_tol=1e-7, rel_tol=1e-7):
                raise RuntimeError('Reloaded selected model changed its validation metrics')
    save_condition_results(output, 'valid', conditions, records, overwrite=True)
    report = {'code_version': CODE_VERSION, 'purpose': 'train', 'method': config['method'],
              'decision_rule': 'logits_argmax', 'class_order': ['negative', 'neutral', 'positive'],
              'neutral_threshold': None, 'selection_protocol': selection_protocol(config['method']),
              'checkpoint_sha256': sha256_file(output / 'best.pt'), 'valid': conditions,
              'training_summary': {'best_epoch': stopper.best_epoch, 'epochs_completed': len(history),
                  'max_epochs': config['epochs'], 'early_stop_patience': config['early_stop_patience'],
                  'stop_reason': stop_reason, 'test_run': False}}
    write_json(output / 'evaluation.json', report)
    print(json.dumps({'best_epoch': stopper.best_epoch, 'valid': selection_score(config['method'], conditions),
                      'results': str(output / 'evaluation.json')}, ensure_ascii=False))


@torch.inference_mode()
def check_forward_agreement(model, batch, method, device):
    model.eval()
    outputs = forward_training(model, batch, device)
    views = [('reference', batch if method == 'M0' else batch['reference'])]
    if method != 'M0':
        views.append(('missing', batch))
    result = {}
    for name, view in views:
        # Physically remove everything except current inputs before prediction.
        inputs = {key: view[key] for key in ('text', 'audio', 'vision', 'audio_lengths', 'vision_lengths')}
        prediction = predict_batch(model, inputs, device)
        for key in ('logits', 'intensity'):
            actual, expected = prediction[key], outputs[key + '_' + name]
            if not torch.allclose(actual, expected, rtol=1e-6, atol=1e-6):
                raise RuntimeError('Training/prediction forward mismatch')
            result[name + '_' + key] = float((actual - expected).abs().max())
    return result


@torch.inference_mode()
def raw_prediction_probe(model, datasets, config, device):
    model.eval()
    result = {}
    for name, dataset in datasets.items():
        batch = next(iter(make_loader(dataset, config)))
        inputs = {key: batch[key] for key in ('text', 'audio', 'vision', 'audio_lengths', 'vision_lengths')}
        result[name] = {key: value.detach().cpu().clone()
                        for key, value in predict_batch(model, inputs, device).items()}
    return result


def smoke(config, output):
    # Fixed bound regardless of formal batch/epoch overrides: five train, three
    # valid, physical sizes [4,1], one update. No formal run or test scoring.
    config = validate_config({**config, 'epochs': 1, 'batch_size': 4,
                              'gradient_accumulation_steps': 2, 'eval_batch_size': 2})
    output = Path(output)
    require_new_output(output)
    device = configure(config, output)
    begin = time.perf_counter()
    store = setup_data(config)
    preprocessing = fit_preprocessing(store.source_path)
    validate_preprocessing(preprocessing, source_sha256=store.binding()['data_sha256'])
    initialize_run(output, config, store, preprocessing, 'smoke')
    indices = _smoke_indices(path_in_project(config['artifact_path']), 4)
    dataset = Subset(training_dataset(store, config['method']), indices)
    valid_datasets = {key: Subset(value, [0, 1, 2]) for key, value in selection_datasets(store, config['method']).items()}
    batches = make_loader(dataset, config, train=True, ordered=True)
    model = build_model(config, ROOT, preprocessing).to(device)
    optimizer = make_optimizer(model, config)
    verify_optimizer(model, optimizer)
    before_heads = {key: getattr(model, key).weight.detach().cpu().clone()
                    for key in ('cls_head', 'post_fusion_layer_3')}
    agreement = [check_forward_agreement(model, batch, config['method'], device) for batch in batches]
    losses, gradients, sizes = train_epoch(model, batches, optimizer, config, device, check_gradients=True)
    if sizes != [4, 1] or losses['optimizer_steps'] != 1:
        raise RuntimeError('Smoke exceeded its intended single update')
    for key, before in before_heads.items():
        if torch.equal(before, getattr(model, key).weight.detach().cpu()):
            raise RuntimeError('Smoke failed to update ' + key)
    conditions, before = evaluate_conditions(model, valid_datasets, config, device)
    before_raw = raw_prediction_probe(model, valid_datasets, config, device)
    temporary = output / 'roundtrip.pt'
    save_checkpoint(temporary, model, config, 1, conditions, store.binding(), ROOT, purpose='smoke')
    checkpoint_hash = sha256_file(temporary)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == 'cuda' else None
    del optimizer, model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    # The real loader must work without refitting or opening a source split.
    from unittest.mock import patch
    with patch('baseline.data.read_pickle', side_effect=AssertionError('Reload touched source data')), \
         patch('standardized.normalization.fit_preprocessing', side_effect=AssertionError('Reload refitted scaler')):
        model = load_for_prediction(temporary, ROOT, device=device, allow_smoke=True)
    if model.av_normalizer.metadata() != preprocessing:
        raise RuntimeError('Reload changed preprocessing')
    after_conditions, after = evaluate_conditions(model, valid_datasets, config, device)
    after_raw = raw_prediction_probe(model, valid_datasets, config, device)
    raw_differences = {}
    for name, values in before_raw.items():
        raw_differences[name] = {}
        for key, value in values.items():
            actual = after_raw[name][key]
            if not torch.allclose(value, actual, rtol=1e-6, atol=1e-6):
                raise RuntimeError('Reload changed raw ' + key)
            raw_differences[name][key] = float((value - actual).abs().max())
    maximum = 0.0
    # Compare all numeric columns so probability field spelling cannot hide a
    # missing classification roundtrip in the evidence.
    for name, rows in before.items():
        for first, second in zip(rows, after[name], strict=True):
            for key in first:
                if isinstance(first[key], (int, float)):
                    delta = abs(first[key] - second[key])
                    maximum = max(maximum, delta)
                    if not math.isclose(first[key], second[key], rel_tol=1e-6, abs_tol=1e-6):
                        raise RuntimeError('Reload changed a prediction column: ' + key)
    save_condition_results(output, 'valid_probe', after_conditions, after)
    temporary.unlink()
    report = {'passed': True, 'purpose': 'smoke', 'code_version': CODE_VERSION,
              'method': config['method'], 'config': config,
              'formal_training_started': False, 'test_evaluated': False, 'valid_probe_n': 3,
              'train_original_indices': indices, 'microbatch_sizes': sizes, 'optimizer_steps': 1,
              'train_loss_terms': losses, 'gradients': gradients, 'forward_agreement': agreement,
              'strict_reload_max_abs_difference': maximum, 'preprocessing_roundtrip_identical': True,
              'strict_reload_raw_max_abs_difference': raw_differences,
              'temporary_checkpoint_sha256': checkpoint_hash, 'temporary_checkpoint_removed': True,
              'peak_cuda_allocated_mb': peak, 'seconds': time.perf_counter() - begin,
              'selection_protocol': selection_protocol(config['method']), 'probe_metrics': after_conditions,
              'note': 'Development probe only; these metrics are not experiment results.'}
    write_json(output / 'smoke_report.json', report)
    print(json.dumps({'passed': True, 'method': config['method'], 'smoke_report': str(output / 'smoke_report.json')}, ensure_ascii=False))


def final_evaluate(output, split, device_override=None, paired=False):
    output = Path(output)
    checkpoint_path = output / 'best.pt'
    checkpoint = load_checkpoint(checkpoint_path, ROOT)
    config = dict(checkpoint['config'])
    binding = checkpoint['data_binding']
    saved_conditions = checkpoint['valid_metrics']
    selected_epoch = checkpoint['epoch']
    del checkpoint
    runtime = {**config, 'device': device_override or config['device']}
    checkpoint_hash = sha256_file(checkpoint_path)
    report_path = output / 'evaluation.json'
    report = read_json(report_path)
    if report.get('checkpoint_sha256') != checkpoint_hash:
        raise ValueError('Evaluation report belongs to another checkpoint')
    device = configure(runtime, output)
    store = setup_data(config)
    if store.binding() != binding:
        raise ValueError('Frozen data binding changed')
    model = load_for_prediction(checkpoint_path, ROOT, device=device)
    conditions, records = evaluate_conditions(model, selection_datasets(store, config['method'], split), runtime, device, split)
    for rows in records.values():
        validate_records(rows, split=split, cohort=store.cohort(split), full=True)
    if split == 'valid':
        for name, metrics in conditions.items():
            for key in ('Accuracy', 'Macro_F1', 'MAE'):
                if not math.isclose(metrics[key], saved_conditions[name][key], abs_tol=1e-7, rel_tol=1e-7):
                    raise ValueError('Validation no longer reproduces selected checkpoint')
    save_condition_results(output, split, conditions, records)
    report[split] = conditions
    if paired:
        panels = store.panels(split)
        canonical = {}
        for condition in dict.fromkeys(c for panel in panels.values() for c in panel['condition_ids']):
            _, rows = evaluate(model, make_loader(store.prediction_dataset(split, condition), runtime), device, split)
            canonical[condition] = {row['index']: row for row in rows}
            save_records(output / 'details' / f'{split}__paired__{condition}.csv', rows)
        report.setdefault('paired', {})[split] = {}
        for name, panel in panels.items():
            scores = {}
            for condition in panel['condition_ids']:
                rows = [canonical[condition][index] for index in panel['indices']]
                if [row['id'] for row in rows] != panel['ids']:
                    raise ValueError('Paired intersection IDs differ from frozen panel')
                scores[condition] = metrics_from_records(rows)
            report['paired'][split][name] = {'cohort_indices': panel['indices'], 'cohort_ids': panel['ids'], 'conditions': scores}
    report['training_summary']['test_run'] |= split == 'test'
    report['training_summary']['best_epoch'] = selected_epoch
    write_json(report_path, report)
    print(json.dumps({'split': split, 'paired': paired, 'results': str(report_path)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('train', 'smoke'):
        entry = sub.add_parser(command)
        entry.add_argument('--method', choices=('M0', 'M1', 'M2'), required=True)
        entry.add_argument('--run-name')
        entry.add_argument('--config', type=Path)
        entry.add_argument('--device', choices=('cpu', 'cuda'))
    entry = sub.add_parser('evaluate')
    entry.add_argument('--run-name', required=True)
    entry.add_argument('--split', choices=('valid', 'test'), required=True)
    entry.add_argument('--paired', action='store_true')
    entry.add_argument('--device', choices=('cpu', 'cuda'))
    args = parser.parse_args()
    if args.run_name and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', args.run_name):
        parser.error('run-name must be a simple name, not a path')
    if args.command == 'evaluate':
        final_evaluate(ROOT / 'outputs' / args.run_name, args.split, args.device, args.paired)
        return
    config = default_config(args.method)
    if args.config:
        overrides = read_json(args.config)
        if 'method' in overrides and overrides['method'] != args.method:
            parser.error('config method conflicts with --method')
        config.update(overrides)
    if args.device:
        config['device'] = args.device
    validate_config(config)
    if args.command == 'train':
        if not args.run_name:
            parser.error('Formal training requires --run-name')
        train(config, ROOT / 'outputs' / args.run_name)
    else:
        if args.run_name:
            parser.error('Smoke only writes a unique development directory')
        output = ROOT / '_dev/records/multitask_smoke' / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + args.method)
        smoke(config, output)


if __name__ == '__main__':
    main()
