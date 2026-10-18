"""Shared data models and JSON (de)serialization. See docs/ARCHITECTURE.md."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class LongformClipsError(Exception):
    """User-facing error. The CLI prints the message without a traceback."""


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    language: str
    duration: float
    segments: list[Segment]
    source: str = ""

    def words(self) -> list[Word]:
        return [w for s in self.segments for w in s.words]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Transcript:
        segs = [
            Segment(
                start=s["start"],
                end=s["end"],
                text=s["text"],
                words=[Word(**w) for w in s.get("words", [])],
            )
            for s in d["segments"]
        ]
        return cls(
            language=d.get("language", "und"),
            duration=d.get("duration", segs[-1].end if segs else 0.0),
            segments=segs,
            source=d.get("source", ""),
        )


@dataclass(frozen=True)
class Sentence:
    """A run of words ending at sentence punctuation or a long pause."""

    start: float
    end: float
    text: str
    words: tuple[Word, ...]


@dataclass(frozen=True)
class Chunk:
    """A group of consecutive sentences (~10-20 s) used as the embedding unit."""

    index: int
    start: float
    end: float
    text: str
    sentence_start: int  # index into the sentence list (inclusive)
    sentence_end: int  # exclusive


@dataclass(frozen=True)
class Boundary:
    """A detected topic shift lying between chunk `after_chunk` and the next one."""

    time: float
    after_chunk: int
    depth: float  # 0..1+, higher = sharper topic change


@dataclass
class Candidate:
    id: int
    start: float
    end: float
    score: float
    reason: str
    text: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise LongformClipsError(f"File not found: {path}") from e
    except json.JSONDecodeError as e:
        raise LongformClipsError(f"Invalid JSON in {path}: {e}") from e


def candidates_to_json(cands: list[Candidate]) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "candidates": [asdict(c) for c in cands]}


def candidates_from_json(d: dict[str, Any]) -> list[Candidate]:
    return [Candidate(**c) for c in d["candidates"]]
