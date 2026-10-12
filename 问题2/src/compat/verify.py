"""Read-only verification of current runtime resources and saved experiments."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from src.runtime import ROOT, resolve_path
from src.compat.provenance import _registry, verify_current_sources


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_layout(root=ROOT):
    root = Path(root)
    directories = {'configs', 'src', 'resources', 'docs', 'outputs'}
    files = {'README.md', 'run.py', 'requirements.txt', 'requirements-lock.txt'}
    missing = sorted(name for name in directories if not (root / name).is_dir())
    missing += sorted(name for name in files if not (root / name).is_file())
    if missing:
        raise ValueError('Required project files are missing: ' + str(missing))
    if (root / 'reproducibility').exists():
        raise ValueError('Historical archives do not belong in the runtime project')
    return {'passed': True, 'entry': 'run.py', 'directories': sorted(directories)}


def verify_resources(root=ROOT):
    root = Path(root).resolve()
    records = _registry(root)['resource_files']
    actual = {path.relative_to(root).as_posix() for path in (root / 'resources').rglob('*')
              if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}
    if actual != set(records):
        raise ValueError('Runtime resource file set changed: ' + str(sorted(actual ^ set(records))))
    for name, record in records.items():
        path = root / name
        if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError('Runtime resource content changed: ' + name)
    sources = verify_current_sources(root)
    return {'passed': True, 'resource_files': len(records), 'source_files': len(sources)}


def verify_checkpoints(store):
    import torch
    from src.compat.provenance import current_provenance, validate_provenance
    result = {}
    for path in sorted((ROOT / 'outputs').rglob('best.pt')):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        method, version = checkpoint.get('method', 'M0'), checkpoint.get('code_version')
        if version == 'four_models_avstd_v1':
            from src.four_model_checkpoint import load_checkpoint
            checkpoint = load_checkpoint(path, ROOT)
        elif version == 'multitask_avstd_v1':
            from src.checkpoint import load_checkpoint
            checkpoint = load_checkpoint(path, ROOT)
        elif version == 'av_standardized_v1':
            from src.models.standardized import validate_checkpoint_preprocessing
            validate_checkpoint_preprocessing(checkpoint)
            validate_provenance(checkpoint['code_sha256'], current_provenance('standardized'), 'standardized')
        elif method in ('M1', 'M2', 'EMT_MISSING_ONLY'):
            group = 'missing_only' if method == 'EMT_MISSING_ONLY' else 'robust'
            validate_provenance(checkpoint['code_sha256'], current_provenance(group), group)
        elif checkpoint.get('data_sha256') != store.binding()['data_sha256']:
            raise ValueError('Legacy data binding mismatch: ' + path.parent.name)
        if 'data_binding' in checkpoint and checkpoint['data_binding'] != store.binding():
            raise ValueError('Frozen data binding mismatch: ' + path.parent.name)
        report_path = path.parent / 'evaluation.json'
        association = 'not_recorded'
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding='utf-8-sig'))
            expected = report.get('checkpoint_sha256')
            if version in ('multitask_avstd_v1', 'four_models_avstd_v1') and not expected:
                raise ValueError('Checkpoint report is missing its weight identity: ' + str(report_path))
            if expected:
                if expected != digest(path):
                    raise ValueError('Checkpoint/evaluation association mismatch: ' + str(path))
                association = 'passed'
        result[path.parent.relative_to(ROOT / 'outputs').as_posix()] = {
            'method': method, 'code_version': version, 'metadata_and_binding': 'passed',
            'evaluation_association': association}
        del checkpoint
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', action='store_true',
                        help='Also verify all saved model metadata, source identities and data bindings')
    args = parser.parse_args(argv)
    layout = verify_layout()
    resources = verify_resources()
    from src.data.missing import MixDataStore
    store = MixDataStore(resolve_path('dataset_construction/artifacts/mix_v1'),
                         resolve_path('data/aligned_50.pkl'))
    environment = {'python': platform.python_version(), 'platform': platform.platform(),
                   'packages': {name: importlib.metadata.version(name) for name in
                                ('torch', 'transformers', 'numpy', 'scipy', 'scikit-learn', 'safetensors')}}
    result = {'passed': True, 'read_only': True, 'layout': layout, 'resources': resources,
              'frozen_data': 'passed', 'environment': environment,
              'scope': 'Current implementation, runtime resources and retained experiments; no historical archives required',
              'checkpoints': verify_checkpoints(store) if args.checkpoints else 'not_checked'}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
