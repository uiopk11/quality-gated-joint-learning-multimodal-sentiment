"""Exact provenance for the standardized implementation, without migrations."""

import hashlib
from pathlib import Path


CODE_VERSION = "av_standardized_v1"


def implementation_provenance(root):
    root = Path(root).resolve()
    paths = [root / name for name in (
        "run_baseline.py", "run_missing.py",
        "run_baseline_standardized.py", "run_missing_standardized.py",
    )]
    for directory in ("baseline", "robust", "standardized", "dataset_construction"):
        paths.extend(sorted((root / directory).glob("*.py")))
    paths.extend(sorted((root / "vendor/EMT-DLFR/models").rglob("*.py")))
    result = {}
    for path in paths:
        with path.open("rb") as stream:
            result[path.relative_to(root).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def validate_implementation(saved, current):
    if not isinstance(saved, dict) or not saved or saved != current:
        raise ValueError("Standardized implementation differs from checkpoint provenance; "
                         "use the matching source version. Legacy models use the original runners.")
