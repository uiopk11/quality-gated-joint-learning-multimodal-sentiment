"""Exact provenance for the standardized implementation, without migrations."""

import hashlib
from pathlib import Path


CODE_VERSION = "av_standardized_v1"


def implementation_provenance(root):
    from src.compat.provenance import current_provenance
    return current_provenance("standardized", root)


def validate_implementation(saved, current):
    from src.compat.provenance import validate_provenance
    validate_provenance(saved, current, "standardized")
