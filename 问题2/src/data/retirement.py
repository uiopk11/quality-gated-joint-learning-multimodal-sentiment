"""Validate the narrowly scoped retirement of the historical A3-like views.

The original accepted manifest remains an immutable experiment identity. This
record permits exactly five known research files to be absent; it does not
make their historical views available or waive any other inventory check.
This module intentionally uses only the standard library so standalone
experiment bundles can carry the same validator.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


RETIREMENT_FILENAME = "exploratory_retirement.json"
RETIRED_FAMILY = "exploratory_a3like"
RETIRED_CONDITION = "a3like_v1_s0"
RETIRED_COUNTS = {"valid": 728, "test": 727}
RETIRED_PATHS = frozenset({
    "exploratory_recipe.json",
    "masks/exploratory_a3like/valid/a3like_v1_s0.npz",
    "masks/exploratory_a3like/test/a3like_v1_s0.npz",
    "metadata/exploratory_a3like/valid/a3like_v1_s0.jsonl",
    "metadata/exploratory_a3like/test/a3like_v1_s0.jsonl",
})
# Only this previously accepted data version has been authorized for retirement.
# A new manifest cannot grant itself an exemption by writing a matching record.
RETIRED_MANIFEST_IDENTITIES = frozenset({(
    "9c8b4a787f7660ceb19b304a66d5f3fd32994dccd089b5f1c117b5dfaf58e9b3",
    "ac192930c154f4589c4114d9b4f7132845a1cd1c24c3fd1a7aba79f3bd888182",
)})


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate retirement JSON key: {key}")
        result[key] = value
    return result


def _fingerprint(value) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def retired_files(root: str | Path, manifest: dict) -> frozenset[str]:
    """Return the exact absent-file exception after validating its full record.

    Without a retirement record, callers must verify every manifest file as
    before. A malformed, partial or inapplicable record always fails closed.
    """
    root = Path(root).resolve()
    record_path = root / RETIREMENT_FILENAME
    if not os.path.lexists(record_path):
        return frozenset()
    if not record_path.is_file() or record_path.is_symlink():
        raise ValueError("Exploratory retirement record must be a regular file")
    try:
        record = json.loads(record_path.read_text(encoding="utf-8-sig"),
                            object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid exploratory retirement record") from exc
    expected_keys = {"format_version", "family", "manifest_sha256",
                     "manifest_fingerprint", "removed_files"}
    if (not isinstance(record, dict) or set(record) != expected_keys
            or type(record["format_version"]) is not int
            or record["format_version"] != 1 or record["family"] != RETIRED_FAMILY):
        raise ValueError("Unsupported exploratory retirement record schema")
    manifest_bytes = (root / "manifest.json").read_bytes()
    actual_identity = (hashlib.sha256(manifest_bytes).hexdigest(), _fingerprint(manifest))
    if (actual_identity not in RETIRED_MANIFEST_IDENTITIES
            or actual_identity != (record["manifest_sha256"], record["manifest_fingerprint"])):
        raise ValueError("Exploratory retirement does not match the authorized historical manifest")
    # Bind the supplied object to the on-disk manifest, not just the record.
    if _fingerprint(json.loads(manifest_bytes.decode("utf-8-sig"))) != actual_identity[1]:
        raise ValueError("Exploratory retirement received a different manifest object")
    removed = record["removed_files"]
    if not isinstance(removed, dict) or set(removed) != RETIRED_PATHS:
        raise ValueError("Exploratory retirement must name exactly the five authorized research files")
    for relative, file_record in removed.items():
        if (not isinstance(file_record, dict) or set(file_record) != {"bytes", "sha256"}
                or type(file_record["bytes"]) is not int or file_record["bytes"] < 0
                or not isinstance(file_record["sha256"], str)
                or file_record != manifest["files"].get(relative)):
            raise ValueError(f"Exploratory retirement inventory mismatch: {relative}")
        if os.path.lexists(root / relative):
            raise ValueError(f"Retired research file is still present: {relative}")
    views = [entry for entry in manifest["views"]
             if entry["evaluation_family"] == RETIRED_FAMILY]
    if len(views) != 2 or {entry["split"] for entry in views} != set(RETIRED_COUNTS):
        raise ValueError("Exploratory retirement requires exactly the historical valid/test views")
    for entry in views:
        split = entry["split"]
        count = RETIRED_COUNTS[split]
        if (entry["condition_id"] != RETIRED_CONDITION or entry["dtype"] != "bool"
                or entry["shape"] != [count, 3, 50] or entry["applicable_count"] != count
                or entry["mask_file"] != f"masks/{RETIRED_FAMILY}/{split}/{RETIRED_CONDITION}.npz"
                or entry["metadata_file"] != f"metadata/{RETIRED_FAMILY}/{split}/{RETIRED_CONDITION}.jsonl"):
            raise ValueError("Retired view differs from the historical all-applicable research cohort")
    for entry in manifest["views"]:
        if entry["evaluation_family"] != RETIRED_FAMILY and (
                entry["mask_file"] in RETIRED_PATHS or entry["metadata_file"] in RETIRED_PATHS):
            raise ValueError("A retained view refers to a retired research file")
    return RETIRED_PATHS
