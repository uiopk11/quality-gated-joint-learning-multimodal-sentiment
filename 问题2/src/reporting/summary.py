"""Inspect or explicitly organize saved evidence; never run an experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.runtime import ROOT
from src.reporting.records import organize_run, summarize_training, write_training_summary


def discover_runs(directory):
    directory = Path(directory)
    if not directory.exists():
        return []
    runs = {path.parent for path in directory.rglob('config.json')
                   if 'reporting_originals' not in path.parts
                   and any((path.parent / name).exists() for name in
                           ('evaluation.json', 'history.json', 'provenance.json', 'details/run.log'))}
    runs.update(path.parent.parent for path in directory.rglob('run.log')
                if path.parent.name == 'details' and 'reporting_originals' not in path.parts)
    return sorted(runs)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else {}


def summarize(run):
    """Compatibility name for callers of the earlier summary-only command."""
    return summarize_training(Path(run))


def compact_split(value):
    if not isinstance(value, dict):
        return {'status': 'not_run', 'metrics': None}
    if 'status' in value and 'Accuracy' not in value:
        return value
    if 'Accuracy' in value:
        return {key: value[key] for key in ('status', 'n', 'Accuracy', 'Macro_F1', 'MAE', 'Pearson', 'Pearson_status') if key in value}
    return {key: compact_split(item) for key, item in value.items() if key != '_reporting'}


def _identity(run, report, training, outputs):
    protocol = report.get('protocol', {})
    selection = training.get('selection_protocol') or report.get('selection_protocol') or {}
    relative = run.relative_to(outputs)
    independent = (training.get('kind') == 'evaluation' or report.get('purpose') == 'evaluation'
                   or 'evaluations' in relative.parts)
    return {
        'run_id': relative.as_posix(),
        'kind': 'evaluation' if independent else 'training',
        'code_version': report.get('code_version') or training.get('code_version')
                        or protocol.get('code_version') or 'unknown',
        'protocol_version': report.get('protocol_version') or protocol.get('protocol_version')
                            or selection.get('protocol_version') or protocol.get('evaluation_version') or 'unknown',
        'method': report.get('method') or training.get('method') or 'unknown',
        'seed': training.get('seed'),
        'best_epoch': None if independent else training.get('best_epoch'),
        'historical': relative.parts[0] == 'old',
    }


def _display_order(item):
    identity = item['identity']
    return (identity['historical'], identity['code_version'] != 'four_models_avstd_v1',
            identity['code_version'], identity['protocol_version'], identity['method'], identity['run_id'])


def _number(value):
    return f'{value:.6f}' if isinstance(value, (int, float)) and not isinstance(value, bool) else 'N/A'


def _table_rows(item):
    identity = item['identity']
    for split in ('valid', 'test'):
        section = item[split]
        if 'Accuracy' in section:
            conditions = [('clean', section)]
        elif 'status' in section:
            conditions = [('-', section)]
        else:
            conditions = sorted(section.items(), key=lambda pair: (pair[0] != 'clean', pair[0]))
        for condition, metric in conditions:
            if not isinstance(metric, dict):
                continue
            yield [identity['run_id'], str(identity['method']), str(identity['best_epoch'] or 'N/A'),
                   split, condition, str(metric.get('n', 'N/A')),
                   *[_number(metric.get(key)) for key in ('Accuracy', 'Macro_F1', 'MAE', 'Pearson')],
                   metric.get('status', 'completed' if 'Accuracy' in metric else 'not_run')]


def _print_table(items, *, discovered, organized, hidden_history, hidden_evaluations):
    print('指标来自已保存记录；不训练、不推理、不重新选模或校准。')
    print('Accuracy / Macro_F1 为 0–1 小数，MAE 为回归误差；不能将不同指标或版本当作同一成绩。')
    print('clean = 不新增人工缺失（保留原生缺口）；main_mixed/mixed_v1_s0 = 冻结混合缺失。valid/test 是各自划分，不是全部数据合并。')
    print('每行是该条件的成绩；Epoch 是保存的最佳轮次，Accuracy 不是 clean/mixed 的选模平均分。')
    if organized:
        print(f'已整理 {discovered} 个记录目录（包括历史及独立评价）；下表按所选范围展示。')
    if hidden_history or hidden_evaluations:
        print(f'默认省略 {hidden_history} 个历史记录、{hidden_evaluations} 个独立评价记录；'
              '分别用 --include-history、--include-evaluations 展开。')
    if not items:
        print('没有符合当前展示范围的记录。历史记录可用 --include-history 查看。')
        return
    groups = {}
    for item in items:
        identity = item['identity']
        groups.setdefault((identity['code_version'], identity['protocol_version']), []).append(item)
    headers = ['Run', 'Method', 'Epoch', 'Split', 'Condition', 'N', 'Accuracy', 'Macro_F1', 'MAE', 'Pearson', 'Status']
    for (version, protocol), group in groups.items():
        print(f'\n版本: {version} | 协议: {protocol}')
        rows = [row for item in group for row in _table_rows(item)]
        widths = [max(len(str(row[index])) for row in [headers, *rows]) for index in range(len(headers))]
        print(' | '.join(value.ljust(width) for value, width in zip(headers, widths)))
        print('-+-'.join('-' * width for width in widths))
        for row in rows:
            print(' | '.join(str(value).ljust(width) for value, width in zip(row, widths)))
    print('\n阅读入口（相对 problem2_solution/）：')
    for item in items:
        identity = item['identity']
        note = '；独立评价，复用已有权重，不算一次新训练' if identity['kind'] == 'evaluation' else ''
        print(f"- {identity['run_id']}{note}")
        if item['analysis']:
            print(f"  分析: outputs/{identity['run_id']}/{item['analysis']}")
        print(f"  指标: {item['evaluation']}")
    print('training.json 是训练摘要；evaluation.json 汇总 valid/test，原始评价与预测来源见其中 artifacts。')
    print('best.pt、config/history/preprocessing/provenance 及 details 原始记录保留；'
          'evaluations/ 是同一权重的独立评价。整理不会补跑缺失实验。')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--run-name', help='Path relative to outputs, including old/<name> or run/evaluations/<id>')
    choice.add_argument('--all', action='store_true', help='Show current training runs; --organize processes every saved record')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--organize', action='store_true', help='Preserve originals, organize predictions/analysis and write v2 reports')
    mode.add_argument('--write-summary', action='store_true', help='Write training.json only, preserving original records')
    parser.add_argument('--include-history', action='store_true', help='Also display archived old/ records, grouped by version')
    parser.add_argument('--include-evaluations', action='store_true', help='Also display independent evaluations, never counted as training runs')
    parser.add_argument('--json', dest='json_output', action='store_true', help='Display compact machine-readable JSON instead of a table')
    parser.add_argument('--full', action='store_true', help='Display complete evaluation documents')
    args = parser.parse_args(argv)
    outputs = (ROOT / 'outputs').resolve()
    if args.all:
        runs = discover_runs(outputs)
    else:
        run = (outputs / args.run_name).resolve()
        if run == outputs or not run.is_relative_to(outputs) or not (run / 'config.json').is_file():
            parser.error('run-name must identify an existing run inside outputs')
        runs = [run]
    entries = []
    comparison_result = None
    try:
        for run in sorted(runs, key=lambda item: (-len(item.parts), str(item))):
            if args.organize:
                report = organize_run(run)
            else:
                report = read(run / 'evaluation.json')
            training = read(run / 'training.json') or summarize_training(run)
            identity = _identity(run, report, training, outputs)
            if args.write_summary and identity['kind'] != 'evaluation':
                training = write_training_summary(run)
                identity = _identity(run, report, training, outputs)
            entries.append({'run': run, 'report': report, 'identity': identity, 'training': training})
        if args.organize:
            from src.reporting.comparisons import build_comparisons
            comparison_result = build_comparisons(outputs)
            # Comparison navigation is refreshed after individual reports.
            for entry in entries:
                entry['report'] = read(entry['run'] / 'evaluation.json')
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
    hidden_history = hidden_evaluations = 0
    displayed = []
    for entry in sorted(entries, key=_display_order):
        identity = entry['identity']
        if args.all and identity['historical'] and not args.include_history:
            hidden_history += 1
            continue
        if args.all and identity['kind'] == 'evaluation' and not args.include_evaluations:
            hidden_evaluations += 1
            continue
        run, report, training = entry['run'], entry['report'], entry['training']
        displayed.append({
                'identity': identity,
                'valid': compact_split(report.get('valid', report.get('summary', {}).get('valid'))),
                'test': compact_split(report.get('test', report.get('summary', {}).get('test'))),
                'training_status': training.get('status'),
                'evaluation': (run / 'evaluation.json').relative_to(ROOT).as_posix(),
                'analysis': report.get('artifacts', {}).get('analysis', {}).get('report'),
                '_full_report': report,
            })
    if args.json_output or args.full:
        result = {item['identity']['run_id']: item['_full_report'] if args.full else
                  {key: value for key, value in item.items() if key != '_full_report'} for item in displayed}
        if comparison_result is not None:
            result['_comparisons'] = comparison_result
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        _print_table(displayed, discovered=len(entries), organized=args.organize,
                     hidden_history=hidden_history, hidden_evaluations=hidden_evaluations)
        if comparison_result is not None:
            print('模型对照: outputs/comparisons/（主结果状态与 robustness_status 分开记录）。')


if __name__ == '__main__':
    main()
