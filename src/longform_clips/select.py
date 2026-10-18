"""Stage 6: candidate generation, scoring and non-overlapping selection."""

from __future__ import annotations

from dataclasses import dataclass

from .models import Boundary, Candidate, Sentence

HOOK_WORDS = {"why", "how", "what", "imagine", "secret", "never", "stop", "biggest", "mistake"}


@dataclass(frozen=True)
class SelectConfig:
    min_seconds: float = 30.0
    max_seconds: float = 60.0
    ideal_seconds: float = 45.0
    boundary_tolerance: float = 3.0  # seconds: how near a boundary counts as "aligned"
    num_candidates: int = 5
    min_candidates: int = 3


def _boundary_strength(t: float, boundaries: list[Boundary], tol: float) -> float:
    best = 0.0
    for b in boundaries:
        if abs(b.time - t) <= tol:
            best = max(best, b.depth * (1.0 - abs(b.time - t) / (tol + 1e-9) * 0.5))
    return best


def score_window(
    sentences: list[Sentence], i: int, j: int, boundaries: list[Boundary], cfg: SelectConfig
) -> tuple[float, str]:
    """Score sentences[i..j] (inclusive). Weights are documented in docs/ARCHITECTURE.md."""
    start, end = sentences[i].start, sentences[j].end
    dur = end - start
    s_start = _boundary_strength(start, boundaries, cfg.boundary_tolerance)
    s_end = _boundary_strength(end, boundaries, cfg.boundary_tolerance)
    n_words = sum(len(s.words) for s in sentences[i : j + 1])
    density = min(n_words / max(dur, 1e-6) / 3.0, 1.0)  # ~3 words/s is brisk speech
    dur_fit = max(0.0, 1.0 - abs(dur - cfg.ideal_seconds) / (cfg.max_seconds - cfg.min_seconds))
    first = sentences[i].text.lower().replace("?", " ").split()
    hook = (
        1.0
        if (sentences[i].text.strip().endswith("?") or (first and first[0] in HOOK_WORDS))
        else 0.0
    )
    score = 0.30 * s_start + 0.25 * s_end + 0.20 * density + 0.15 * dur_fit + 0.10 * hook
    reason = (
        f"start_boundary={s_start:.2f} end_boundary={s_end:.2f} "
        f"speech_density={density:.2f} duration_fit={dur_fit:.2f} hook={hook:.0f}"
    )
    return round(score, 4), reason


def generate_windows(sentences: list[Sentence], cfg: SelectConfig) -> list[tuple[int, int]]:
    """All sentence-aligned (i, j) windows with min_seconds <= duration <= max_seconds."""
    out = []
    for i in range(len(sentences)):
        for j in range(i, len(sentences)):
            dur = sentences[j].end - sentences[i].start
            if dur > cfg.max_seconds:
                break
            if dur >= cfg.min_seconds:
                out.append((i, j))
    return out


def select_candidates(
    sentences: list[Sentence], boundaries: list[Boundary], cfg: SelectConfig | None = None
) -> list[Candidate]:
    """Pick up to `num_candidates` non-overlapping windows, best score first.

    Deterministic: ties break on earlier start. Result ids are 1..N ordered by rank.
    """
    cfg = cfg or SelectConfig()
    scored = []
    for i, j in generate_windows(sentences, cfg):
        sc, reason = score_window(sentences, i, j, boundaries, cfg)
        scored.append((-sc, sentences[i].start, i, j, reason))
    scored.sort()
    chosen: list[Candidate] = []
    for neg, _, i, j, reason in scored:
        if len(chosen) >= cfg.num_candidates:
            break
        s, e = sentences[i].start, sentences[j].end
        if any(s < c.end and e > c.start for c in chosen):
            continue
        text = " ".join(x.text for x in sentences[i : j + 1])
        chosen.append(Candidate(len(chosen) + 1, s, e, -neg, reason, text))
    return chosen
