"""Current source integrity and exact identities of retained experiments.

Historical checkpoints and the accepted data manifest keep their original
source dictionaries. Their complete identities are registered once; loading
them does not require copies of previous source trees. Newly saved evidence
always contains hashes of the actual current files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


GROUPS = ("standardized", "multitask", "robust", "missing_only", "construction", "four_models")
REGISTRY_PATH = "configs/runtime_integrity.json"


def _root(root=None):
    return Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _inside(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Integrity paths must be nonempty package-relative paths")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Integrity path escapes package: " + relative)
    return path


def _valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _fingerprint(saved):
    if not isinstance(saved, dict) or not saved or any(
            not isinstance(name, str) or not name or not _valid_digest(value)
            for name, value in saved.items()):
        raise ValueError("Invalid implementation provenance")
    content = json.dumps(saved, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def controlled_paths(root=None):
    """Current source/config files; the registry does not hash itself."""
    root = _root(root)
    files = set(root.glob("*.py"))
    files.update((root / "src").rglob("*.py"))
    files.update((root / "configs").rglob("*.json"))
    files.update((root / "resources/third_party").rglob("*.py"))
    files.discard(root / REGISTRY_PATH)
    return sorted(path.relative_to(root).as_posix() for path in files if path.is_file())


def _registry(root=None):
    root = _root(root)
    with (root / REGISTRY_PATH).open(encoding="utf-8-sig") as stream:
        registry = json.load(stream)
    fields = {"format_version", "current_files", "legacy_fingerprints", "paths",
              "tokenizer_mapping", "source", "resource_files"}
    if (not isinstance(registry, dict) or set(registry) != fields
            or registry.get("format_version") != 1
            or not isinstance(registry.get("current_files"), dict)
            or not registry["current_files"]
            or any(not _valid_digest(value) for value in registry["current_files"].values())
            or not isinstance(registry.get("legacy_fingerprints"), dict)
            or set(registry["legacy_fingerprints"]) != set(GROUPS)
            or any(not isinstance(values, list) or any(not _valid_digest(value) for value in values)
                   or len(values) != len(set(values))
                   for values in registry["legacy_fingerprints"].values())
            or not all(isinstance(registry.get(name), dict) for name in
                       ("paths", "tokenizer_mapping", "source", "resource_files"))):
        raise ValueError("Invalid runtime integrity registry")
    return registry


def _actual_files(root, relatives):
    return {relative: _hash(_inside(root, relative)) for relative in sorted(relatives)}


def _verify_registry(root, registry):
    expected = registry["current_files"]
    if set(controlled_paths(root)) != set(expected):
        raise ValueError("Registered implementation file set changed")
    actual = _actual_files(root, expected)
    if actual != expected:
        raise ValueError("Registered implementation content changed")
    return actual


def verify_current_sources(root=None):
    root = _root(root)
    return _verify_registry(root, _registry(root))


def current_provenance(group, root=None):
    """Hash the actual shared runtime, keeping version identity in the payload."""
    if group not in GROUPS:
        raise ValueError("Unknown implementation group: " + str(group))
    return verify_current_sources(root)


def validate_provenance(saved, current, group, root=None):
    if group not in GROUPS:
        raise ValueError("Unknown implementation group: " + str(group))
    identity = _fingerprint(saved)
    root = _root(root)
    registry = _registry(root)
    actual = _verify_registry(root, registry)
    if current != actual:
        raise ValueError("Supplied current provenance does not match actual source")
    if saved != actual and identity not in registry["legacy_fingerprints"][group]:
        raise ValueError(f"Unregistered {group} checkpoint/source provenance")


def validate_four_model_provenance(saved, current, root=None):
    validate_provenance(saved, current, "four_models", root)


def validate_frozen_code(saved, root=None):
    validate_provenance(saved, current_provenance("construction", root), "construction", root)


def validate_tokenizer(saved, current, root=None):
    """Validate the exact vocabulary and token IDs, allowing its known move."""
    if not isinstance(saved, dict) or not isinstance(current, dict):
        raise ValueError("Invalid tokenizer identity")
    root = _root(root)
    registry = _registry(root)
    mapping = registry["tokenizer_mapping"]
    expected = mapping.get("saved")
    relative = mapping.get("current_relative_path")
    if not isinstance(expected, dict) or not isinstance(relative, str):
        raise ValueError("Missing tokenizer relocation identity")
    target = _inside(root, relative)
    if current.get("vocab_path") != str(target) or current.get("vocab_sha256") != _hash(target):
        raise ValueError("Tokenizer must use the registered package vocabulary")
    without_path = lambda value: {key: item for key, item in value.items() if key != "vocab_path"}
    if without_path(saved) != without_path(current):
        raise ValueError("Tokenizer content or special IDs changed")
    if saved != current and saved != expected:
        raise ValueError("Unregistered vocabulary relocation")
    _verify_registry(root, registry)
