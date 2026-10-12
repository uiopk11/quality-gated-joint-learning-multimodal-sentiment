"""Regenerate all Q3 cursor_draw paper figures."""

from __future__ import annotations

import runpy
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
SCRIPTS = [
    "draw_q3_framework.py",
    "draw_q3_fixed_target_mask.py",
    "draw_q3_shapley_exact.py",
    "draw_q3_multiscale_local.py",
    "draw_q3_evidence_trace.py",
]


def main() -> None:
    for name in SCRIPTS:
        path = CODE_DIR / name
        print(f"==> {name}")
        runpy.run_path(str(path), run_name="__main__")


if __name__ == "__main__":
    main()
