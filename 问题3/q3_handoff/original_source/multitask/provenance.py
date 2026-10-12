"""New exact fingerprints extending, but never changing, legacy provenance."""
import hashlib
from pathlib import Path

from multitask.config import CODE_VERSION, PROTOCOL_VERSION
from standardized.provenance import implementation_provenance as standardized_provenance


def implementation_provenance(root):
    root = Path(root).resolve()
    result = standardized_provenance(root)
    paths = [root / "run_multitask.py", root / "configs/multitask_v1.json",
             *sorted((root / "multitask").glob("*.py"))]
    for path in paths:
        with path.open("rb") as stream:
            result[path.relative_to(root).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def validate_implementation(saved, current):
    if not isinstance(saved, dict) or not saved or saved != current:
        raise ValueError("Multitask implementation differs from checkpoint provenance; "
                         "use the matching source version and its original entrypoint")
