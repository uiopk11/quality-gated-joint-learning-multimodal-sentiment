"""External, read-only adapters for the formal complete-input M0 predictor.

Configure writable runtime paths before any project or ML dependency is imported.
The original_source directory is an unchanged copy; customization lives here.
"""
from .runtime import configure_runtime

HANDOFF_ROOT = configure_runtime()

__all__ = ["HANDOFF_ROOT", "configure_runtime"]
