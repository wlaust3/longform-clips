"""Glue between stages. Kept separate from the CLI so it is testable."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .boundaries import detect_boundaries
from .captions import build_ass, words_to_cues
from .chunk import make_chunks, split_sentences
from .embed import Embedder
from .models import Candidate, Transcript
from .render import build_ffmpeg_cmd, render_clip
from .select import SelectConfig, select_candidates


def find_candidates(
    transcript: Transcript, embedder: Embedder, cfg: SelectConfig | None = None, seed: int = 0
) -> list[Candidate]:
    np.random.seed(seed)
    sentences = split_sentences(transcript)
    chunks = make_chunks(sentences)
    vectors = embedder.embed([c.text for c in chunks]) if chunks else np.zeros((0, 1))
    boundaries = detect_boundaries(chunks, vectors)
    return select_candidates(sentences, boundaries, cfg)


def render_candidate(
    transcript: Transcript,
    cand: Candidate,
    src: Path,
    out: Path,
    width: int,
    height: int,
    crf: int = 20,
    preset: str = "medium",
) -> Path:
    """Write <out>.ass next to the output and render the clip."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cues = words_to_cues(transcript.words(), cand.start, cand.end)
    ass = out.with_suffix(".ass")
    ass.write_text(build_ass(cues, width, height), encoding="utf-8")
    render_clip(build_ffmpeg_cmd(src, cand.start, cand.end, ass, out, width, height, crf, preset))
    return out
