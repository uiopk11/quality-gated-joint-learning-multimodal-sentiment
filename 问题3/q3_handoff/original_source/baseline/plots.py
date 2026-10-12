"""Dependency-free SVG export for the three-class confusion matrix.

AI-assisted implementation: OpenAI Codex, 2026-09-24. Numerical matrices
remain available as CSV and JSON for plotting with other scientific tools.
"""
from html import escape
from pathlib import Path

import numpy as np


def write_confusion_svg(path, matrix, *, normalized=False, title="Confusion matrix"):
    """Export a vector heatmap with rows=true and columns=predicted classes."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (3, 3) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected a finite nonnegative 3 by 3 matrix")
    if normalized and (values > 1).any():
        raise ValueError("Normalized confusion values must be in [0, 1]")
    maximum = 1.0 if normalized else max(float(values.max()), 1.0)
    names = ("Negative", "Neutral", "Positive")
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="560" viewBox="0 0 640 560" role="img">',
        f'<title>{escape(title)}</title>',
        '<desc>Rows are true classes; columns are predicted classes. Class order: Negative, Neutral, Positive.</desc>',
        '<rect width="640" height="560" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#172b4d">',
        f'<text x="320" y="38" text-anchor="middle" font-size="21">{escape(title)}</text>',
        '<text x="365" y="79" text-anchor="middle" font-size="17">Predicted class</text>',
        '<text x="30" y="310" text-anchor="middle" font-size="17" transform="rotate(-90 30 310)">True class</text>',
    ]
    for j, name in enumerate(names):
        parts.append(f'<text x="{235 + j * 130}" y="119" text-anchor="middle" font-size="16">{name}</text>')
    for i, name in enumerate(names):
        parts.append(f'<text x="153" y="{209 + i * 120}" text-anchor="end" font-size="16">{name}</text>')
        for j in range(3):
            value = values[i, j]
            fraction = float(value / maximum)
            rgb = tuple(round(a + fraction * (b - a)) for a, b in zip((239, 246, 255), (30, 64, 175)))
            color = "#{:02x}{:02x}{:02x}".format(*rgb)
            foreground = "white" if fraction >= 0.55 else "#172b4d"
            label = f"{value:.1%}" if normalized else f"{value:g}"
            parts.append(f'<rect x="{170 + j * 130}" y="{140 + i * 120}" width="130" height="120" fill="{color}" stroke="white" stroke-width="2"/>')
            parts.append(f'<text x="{235 + j * 130}" y="{210 + i * 120}" text-anchor="middle" font-size="24" fill="{foreground}">{label}</text>')
    caption = "Row-normalized proportions; empty true-class rows are zero." if normalized else "Counts; all samples are retained."
    parts.extend([f'<text x="320" y="535" text-anchor="middle" font-size="13">{caption}</text>', '</g></svg>'])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
