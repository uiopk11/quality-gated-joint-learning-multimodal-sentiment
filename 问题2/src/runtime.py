"""Package-local resources and the few paths saved in existing experiments."""
from pathlib import Path
import os
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def _metadata(root=ROOT):
    from src.compat.provenance import _registry
    return _registry(root)


def configure_environment():
    """Keep transient library caches outside the delivery tree; remain offline."""
    cache = Path(tempfile.gettempdir()) / 'problem2_runtime_cache' / 'huggingface'
    os.environ['HF_HOME'] = str(cache)
    os.environ['HF_HUB_CACHE'] = str(cache / 'hub')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def resolve_path(value, root=None):
    root = Path(root).resolve() if root is not None else ROOT
    raw = str(value).replace('\\', '/')
    paths = _metadata(root)['paths']
    prefixes = [root.as_posix().rstrip('/'), *paths['original_roots']]
    for prefix in prefixes:
        prefix = prefix.replace('\\', '/').rstrip('/')
        if raw.lower().startswith(prefix.lower() + '/'):
            raw = raw[len(prefix) + 1:]
            break
    for before, after in paths['aliases'].items():
        if raw == before or raw.startswith(before + '/'):
            raw = after + raw[len(before):]
            break
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def source_path(relative, root=None):
    root = Path(root).resolve() if root is not None else ROOT
    mapping = _metadata(root)['paths']['source_map']
    return root / mapping.get(relative.replace('\\', '/'), relative)


def source_commit():
    """The unchanged upstream EMT-DLFR commit recorded by existing runs."""
    return _metadata()['source']['commit']


def protected_record_path(record):
    """Locate a current construction input; callers verify its bytes/hash."""
    return resolve_path(record['path'])
