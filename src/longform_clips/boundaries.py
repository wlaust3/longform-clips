"""Stage 5: topic-shift detection (TextTiling-style valley/depth scoring on embeddings)."""

from __future__ import annotations

import numpy as np

from .models import Boundary, Chunk


def adjacent_similarity(vectors: np.ndarray, window: int = 2) -> np.ndarray:
    """Cosine similarity across each gap between chunks.

    Gap g sits between chunk g and g+1. Each side is the mean of up to `window` chunks, which
    smooths single-chunk noise. Returns an array of length n-1 (empty if n < 2).
    """
    n = len(vectors)
    sims = np.zeros(max(n - 1, 0))
    for g in range(n - 1):
        left = vectors[max(0, g - window + 1) : g + 1].mean(axis=0)
        right = vectors[g + 1 : g + 1 + window].mean(axis=0)
        denom = np.linalg.norm(left) * np.linalg.norm(right)
        sims[g] = float(left @ right / denom) if denom else 0.0
    return sims


def depth_scores(sims: np.ndarray) -> np.ndarray:
    """TextTiling depth: how far each gap's similarity sits below the peaks on both sides."""
    depths = np.zeros_like(sims)
    for g in range(len(sims)):
        left = g
        while left > 0 and sims[left - 1] >= sims[left]:
            left -= 1
        right = g
        while right < len(sims) - 1 and sims[right + 1] >= sims[right]:
            right += 1
        depths[g] = (sims[left] - sims[g]) + (sims[right] - sims[g])
    return depths


def detect_boundaries(
    chunks: list[Chunk], vectors: np.ndarray, window: int = 2, cutoff_std: float = 0.5
) -> list[Boundary]:
    """Return boundaries at local depth maxima whose depth exceeds mean + cutoff_std * std.

    Depth values are normalised by the maximum so `Boundary.depth` is in 0..1.
    """
    if len(chunks) < 3:
        return []
    sims = adjacent_similarity(vectors, window)
    depths = depth_scores(sims)
    if depths.max() <= 0:
        return []
    cutoff = depths.mean() + cutoff_std * depths.std()
    norm = depths / depths.max()
    out: list[Boundary] = []
    for g, d in enumerate(depths):
        if d <= 0 or d < cutoff:
            continue
        lo = depths[g - 1] if g > 0 else -1.0
        hi = depths[g + 1] if g + 1 < len(depths) else -1.0
        if d >= lo and d > hi:
            out.append(Boundary(time=chunks[g].end, after_chunk=g, depth=float(norm[g])))
    return out
