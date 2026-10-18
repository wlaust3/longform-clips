"""Stage 3: words -> sentences -> chunks (the embedding unit)."""

from __future__ import annotations

from .models import Chunk, Sentence, Transcript, Word

SENTENCE_END = (".", "?", "!", "。", "？", "！", "…")


def split_sentences(
    transcript: Transcript, max_gap: float = 1.0, max_sentence_seconds: float = 20.0
) -> list[Sentence]:
    """Group word-timestamped text into sentences.

    A sentence ends at terminal punctuation, a pause longer than `max_gap` seconds, or when it
    would exceed `max_sentence_seconds` (guards against unpunctuated rambling).
    """
    sentences: list[Sentence] = []
    cur: list[Word] = []

    def flush() -> None:
        if cur:
            text = " ".join(w.text.strip() for w in cur).strip()
            sentences.append(Sentence(cur[0].start, cur[-1].end, text, tuple(cur)))
            cur.clear()

    for w in transcript.words():
        if cur and (w.start - cur[-1].end > max_gap or w.end - cur[0].start > max_sentence_seconds):
            flush()
        cur.append(w)
        if w.text.strip().endswith(SENTENCE_END):
            flush()
    flush()
    return sentences


def make_chunks(sentences: list[Sentence], target_seconds: float = 12.0) -> list[Chunk]:
    """Greedily pack consecutive sentences into chunks of roughly `target_seconds`."""
    chunks: list[Chunk] = []
    i = 0
    n = len(sentences)
    while i < n:
        j = i + 1
        while j < n and sentences[j].end - sentences[i].start <= target_seconds:
            j += 1
        text = " ".join(s.text for s in sentences[i:j])
        chunks.append(Chunk(len(chunks), sentences[i].start, sentences[j - 1].end, text, i, j))
        i = j
    return chunks
