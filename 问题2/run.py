"""Single offline entry point. Commands dispatch lazily without sampling RNG."""
import argparse
import importlib
import sys

sys.dont_write_bytecode = True


VERSIONS = {
    'four-models': 'src.training.four_models',
    'multitask': 'src.training.multitask',
    'baseline': 'src.training.baseline',
    'standardized': 'src.training.baseline_standardized',
    'missing': 'src.training.missing',
    'missing-standardized': 'src.training.missing_standardized',
    'missing-only': 'src.training.missing_only',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('train', 'evaluate'):
        entry = sub.add_parser(command, add_help=False)
        entry.add_argument('--version', choices=VERSIONS, default='multitask')
    sub.add_parser('verify', add_help=False)
    sub.add_parser('report', add_help=False)
    sub.add_parser('data', add_help=False)
    sub.add_parser('predict', add_help=False, help='Predict all unlabeled Attachment 3 samples')
    sub.add_parser('legacy', add_help=False)
    args, rest = parser.parse_known_args()
    if args.command == 'predict':
        from src.predict_attachment3 import main as predict
        predict(rest)
        return
    if args.command in ('train', 'evaluate'):
        command = args.command
        if command == 'evaluate' and args.version not in ('multitask', 'four-models'):
            # Historical final test retains its frozen regression threshold.
            if '--split' in rest:
                i = rest.index('--split')
                if i + 1 >= len(rest) or rest[i + 1] != 'test':
                    parser.error('历史回归 evaluate 支持 --split test；已有 valid 报告用 report 查看。')
                del rest[i:i + 2]
            command = 'test'
        sys.argv = [sys.argv[0], command, *rest]
        importlib.import_module(VERSIONS[args.version]).main()
    elif args.command == 'legacy':
        legacy = argparse.ArgumentParser(description='Explicit original protocol entry')
        versions = VERSIONS
        legacy.add_argument('--version', required=True, choices=versions)
        selected, forwarded = legacy.parse_known_args(rest)
        if forwarded and forwarded[0] == 'smoke':
            parser.error('smoke 是开发验证；请从工程外测试脚本调用，并显式指定开发输出目录。')
        sys.argv = [sys.argv[0], *forwarded]
        importlib.import_module(versions[selected.version]).main()
    elif args.command == 'verify':
        from src.compat.verify import main as verify
        verify(rest)
    elif args.command == 'report':
        from src.reporting.summary import main as report
        report(rest)
    else:
        # Original verify refreshes acceptance, so expose it with an explicit
        # name; the ordinary verification command is read-only.
        if rest and rest[0] == 'verify':
            parser.error('只读检查请用 run.py verify；刷新验收须显式使用 data refresh-acceptance。')
        if rest and rest[0] == 'refresh-acceptance':
            rest[0] = 'verify'
        sys.argv = [sys.argv[0], *rest]
        from src.data.construction.cli import main as data
        data()


if __name__ == '__main__':
    main()
