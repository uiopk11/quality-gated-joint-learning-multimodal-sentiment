"""New exact fingerprints extending, but never changing, legacy provenance."""
import hashlib
from pathlib import Path

from src.training.config import CODE_VERSION, PROTOCOL_VERSION
from src.compat.standardized import implementation_provenance as standardized_provenance


def implementation_provenance(root):
    from src.compat.provenance import current_provenance
    return current_provenance("multitask", root)


def validate_implementation(saved, current):
    from src.compat.provenance import validate_provenance
    validate_provenance(saved, current, "multitask")
