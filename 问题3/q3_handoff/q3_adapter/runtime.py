"""Contain all writable runtime state inside the handoff, before ML imports."""
import os
from pathlib import Path
import sys
import tempfile

def configure_runtime():
    root = Path(__file__).resolve().parents[1]
    runtime = root / 'q3_adapter' / '_runtime'
    variables = {
        'TMP': 'tmp', 'TEMP': 'tmp', 'TMPDIR': 'tmp',
        'HF_HOME': 'huggingface', 'HF_HUB_CACHE': 'huggingface/hub',
        'TRANSFORMERS_CACHE': 'huggingface/transformers',
        'XDG_CACHE_HOME': 'cache', 'TORCH_HOME': 'torch',
        'TORCH_EXTENSIONS_DIR': 'torch_extensions',
        'TORCHINDUCTOR_CACHE_DIR': 'torchinductor', 'TRITON_CACHE_DIR': 'triton',
        'CUDA_CACHE_PATH': 'cuda', 'MPLCONFIGDIR': 'matplotlib',
        'PYTHONPYCACHEPREFIX': 'pycache', 'NUMBA_CACHE_DIR': 'numba',
    }
    for name, relative in variables.items():
        folder = runtime / relative
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(folder)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    sys.dont_write_bytecode = True
    tempfile.tempdir = str(runtime / 'tmp')
    # Prevent relative logs/outputs from reaching the caller's project directory.
    os.chdir(root)
    source = str(root / 'original_source')
    if source not in sys.path:
        sys.path.insert(0, source)
    return root
