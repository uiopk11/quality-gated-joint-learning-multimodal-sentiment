"""Verify the saved repository snapshot without importing ML dependencies."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def verify():
    errors = []
    manifest = ROOT / 'docs/provenance/文件清单与SHA256.csv'
    records = read_csv(manifest)
    for row in records:
        path = ROOT / row['path']
        if not path.is_file():
            errors.append('Missing: ' + row['path'])
        elif path.stat().st_size != int(row['bytes']) or digest(path) != row['sha256']:
            errors.append('Content changed: ' + row['path'])

    python_files = sorted(p for p in ROOT.rglob('*.py')
                          if not {'.git', '__pycache__', '_runtime'}.intersection(p.parts))
    for path in python_files:
        try:
            ast.parse(path.read_bytes(), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f'Python syntax: {path.relative_to(ROOT)}: {exc}')

    a3 = read_csv(ROOT / '问题2/附件3_测试集预测结果.csv')
    expected_a3 = [f'附件3_{i:02d}' for i in range(1, 31)]
    if [r['sample_id'] for r in a3] != expected_a3:
        errors.append('Attachment3 sample coverage/order mismatch')
    a4 = read_csv(ROOT / '问题3/附件4_测试集预测与解释结果.csv')
    if len(a4) != 20 or len(a4[0]) != 104:
        errors.append('Attachment4 shape mismatch')
    if [r['sample_id'] for r in a4] != [f'{i:02d}' for i in range(1, 21)]:
        errors.append('Attachment4 sample coverage/order mismatch')
    frames = set()
    for row in a4:
        for key, value in row.items():
            if key.endswith('keyframe_image') and value:
                frames.add(value)
                if not (ROOT / '问题3' / value).is_file():
                    errors.append('Missing keyframe: ' + value)

    q2 = ROOT / '问题2'
    registry = json.loads((q2 / 'configs/runtime_integrity.json').read_text(encoding='utf-8-sig'))
    for name, expected in registry['current_files'].items():
        path = q2 / name
        if not path.is_file() or digest(path) != expected:
            errors.append('Registered source fingerprint: ' + name)
    # External data and weights are intentionally supplied separately.
    external = []
    for name, expected in registry['resource_files'].items():
        path = q2 / name
        if not path.is_file():
            if name.startswith('resources/data/') or name.endswith('model.safetensors'):
                external.append(name)
            else:
                errors.append('Registered resource missing: ' + name)
        elif path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            errors.append('Registered resource fingerprint: ' + name)

    prohibited = {'.pt', '.pth', '.safetensors', '.exe', '.dll', '.zip', '.rar', '.mp4', '.wav'}
    large_files = []
    for path in ROOT.rglob('*'):
        if path.is_file() and '.git' not in path.parts:
            if path.suffix.lower() in prohibited:
                errors.append('External artifact unexpectedly bundled: ' + str(path.relative_to(ROOT)))
            if path.stat().st_size >= 100 * 1024 * 1024:
                large_files.append(str(path.relative_to(ROOT)))
    if large_files:
        errors.append('Files exceed GitHub single-file limit: ' + str(large_files))
    return {'passed': not errors, 'manifest_files': len(records), 'python_files': len(python_files),
            'attachment3_rows': len(a3), 'attachment4_rows': len(a4), 'attachment4_columns': len(a4[0]),
            'referenced_keyframes': len(frames), 'external_q2_resources': len(external),
            'model_execution_performed': False, 'errors': errors}


if __name__ == '__main__':
    report = verify()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0 if report['passed'] else 1)
