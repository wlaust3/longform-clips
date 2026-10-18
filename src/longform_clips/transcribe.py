"""Stage 2: speech-to-text with word timestamps (faster-whisper, lazy-imported)."""

from __future__ import annotations

from pathlib import Path

from .models import LongformClipsError, Segment, Transcript, Word


def transcribe(
    media: Path,
    model_size: str = "small",
    device: str = "auto",
    compute_type: str = "default",
    language: str | None = None,
) -> Transcript:
    """Transcribe `media` locally. Models download on first use (see README)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise LongformClipsError(
            "faster-whisper is not installed. Run: pip install -e '.[ml]', "
            "or pass --transcript to reuse an existing transcript.json."
        ) from e
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments_iter, info = model.transcribe(
        str(media),
        word_timestamps=True,
        vad_filter=True,
        language=language,
        beam_size=5,
        temperature=0.0,  # deterministic decoding
    )
    segments: list[Segment] = []
    for seg in segments_iter:
        words = [Word(float(w.start), float(w.end), w.word.strip()) for w in (seg.words or [])]
        segments.append(Segment(float(seg.start), float(seg.end), seg.text.strip(), words))
    return Transcript(
        language=info.language, duration=float(info.duration), segments=segments, source=str(media)
    )
