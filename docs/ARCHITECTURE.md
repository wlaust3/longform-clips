# Architecture

Longform Clips is a linear pipeline. Each stage is a pure-ish function in its own module, so it
can be unit-tested without models, network or media. Heavy dependencies (`faster-whisper`,
`sentence-transformers`) are imported lazily inside the functions that need them.

```
video/URL -> ingest -> transcribe -> chunk -> embed -> boundaries -> select -> captions -> render
```

Package: `src/longform_clips/`. CLI entry point: `longform-clips` (`longform_clips.cli:app`).

## Pipeline stages

| # | Module | Responsibility |
|---|---|---|
| 1 | `ingest.py` | Resolve a local path or download a URL with yt-dlp; check ffmpeg/ffprobe; probe duration |
| 2 | `transcribe.py` | faster-whisper, word timestamps, VAD, `temperature=0` for determinism |
| 3 | `chunk.py` | Words -> sentences (punctuation, pause > 1 s, or 20 s cap) -> chunks of ~12 s |
| 4 | `embed.py` | `Embedder` protocol; `SentenceTransformerEmbedder` (MiniLM), `HashingEmbedder` (offline) |
| 5 | `boundaries.py` | Cosine similarity across each chunk gap (window of 2), TextTiling depth scores, peak picking |
| 6 | `select.py` | Enumerate sentence-aligned 30-60 s windows, score, greedily pick non-overlapping top N |
| 7 | `captions.py` | Words in clip -> short cues (<= 4 words / 28 chars, break at punctuation) -> ASS text |
| 8 | `render.py` | FFmpeg: cut, center-crop to 9:16, scale to target size, burn ASS via `subtitles` filter |
| - | `pipeline.py` | Glue: `find_candidates(...)`, `render_candidate(...)` |
| - | `preview_html.py` | Static `preview.html` with relative `<video>` links |
| - | `models.py` | Dataclasses, JSON helpers, `LongformClipsError` |
| - | `cli.py` | Typer commands `analyze`, `preview`, `export` |

## Output directory layout

```
out/
  source/source.mp4        # only when input was a URL
  source.json              # {"path": "<absolute path to the source video>"}
  transcript.json
  candidates.json
  previews/candidate_N.mp4 # 360x640, plus candidate_N.ass
  preview.html
  clips/clip_N.mp4         # 1080x1920, plus clip_N.ass
```

## Data formats

All JSON files carry `"schema_version": 1`. Times are seconds (float) from the start of the
source video.

### transcript.json

```json
{
  "schema_version": 1,
  "language": "en",
  "duration": 1834.2,
  "source": "/abs/path/video.mp4",
  "segments": [
    {
      "start": 0.0, "end": 4.2, "text": "Welcome to the show.",
      "words": [{"start": 0.0, "end": 0.4, "text": "Welcome"}, {"start": 0.4, "end": 0.6, "text": "to"}]
    }
  ]
}
```

### Chunks (in memory; may be dumped with `--debug` later)

```json
{"index": 3, "start": 36.0, "end": 48.5, "text": "...", "sentence_start": 9, "sentence_end": 12}
```

`sentence_start` is inclusive and `sentence_end` is exclusive, indexing the sentence list.

### Boundaries (in memory)

```json
{"time": 48.5, "after_chunk": 3, "depth": 0.82}
```

`depth` is normalised to 0..1 (1 = sharpest topic shift in this video).

### candidates.json

```json
{
  "schema_version": 1,
  "candidates": [
    {
      "id": 1, "start": 312.4, "end": 355.9, "score": 0.7123,
      "reason": "start_boundary=0.91 end_boundary=0.62 speech_density=0.80 duration_fit=0.97 hook=1",
      "text": "full transcript text of the clip"
    }
  ]
}
```

Rules: 30 <= `end - start` <= 60; `start` is a sentence start and `end` a sentence end; candidates
do not overlap; ids are 1..N by descending score.

## Scoring (milestone 1)

`score = 0.30*start_boundary + 0.25*end_boundary + 0.20*speech_density + 0.15*duration_fit + 0.10*hook`

- `start_boundary` / `end_boundary`: depth of the nearest topic boundary within 3 s (0 if none)
- `speech_density`: words per second, capped at 3 w/s, scaled to 0..1
- `duration_fit`: 1 at 45 s, falling linearly toward the 30/60 s limits
- `hook`: 1 if the first sentence is a question or starts with a hook word, else 0

Ties break on earlier start time, so results are deterministic. An LLM scorer (later milestone)
should plug in as an alternative to `score_window` without changing `Candidate`.

## Module interfaces

```python
# models.py
Word(start, end, text); Segment(start, end, text, words); Transcript(language, duration, segments, source)
Sentence(start, end, text, words); Chunk(index, start, end, text, sentence_start, sentence_end)
Boundary(time, after_chunk, depth); Candidate(id, start, end, score, reason, text)
class LongformClipsError(Exception)   # user-facing; CLI prints "error: ..." and exits 1

# ingest.py
is_url(s) -> bool;  require_ffmpeg() -> None;  probe_duration(path) -> float
download(url, dest_dir) -> Path;  resolve_input(src, out_dir) -> Path

# transcribe.py
transcribe(media, model_size="small", device="auto", compute_type="default", language=None) -> Transcript

# chunk.py
split_sentences(transcript, max_gap=1.0, max_sentence_seconds=20.0) -> list[Sentence]
make_chunks(sentences, target_seconds=12.0) -> list[Chunk]

# embed.py
class Embedder(Protocol): embed(texts: Sequence[str]) -> np.ndarray   # (n, d), L2-normalised
get_embedder(name) -> Embedder

# boundaries.py
adjacent_similarity(vectors, window=2) -> ndarray; depth_scores(sims) -> ndarray
detect_boundaries(chunks, vectors, window=2, cutoff_std=0.5) -> list[Boundary]

# select.py
SelectConfig(min_seconds=30, max_seconds=60, ideal_seconds=45, boundary_tolerance=3, num_candidates=5, min_candidates=3)
select_candidates(sentences, boundaries, cfg=None) -> list[Candidate]

# captions.py
words_to_cues(words, clip_start, clip_end, max_words=4, max_chars=28) -> list[Cue]
build_ass(cues, width=1080, height=1920, font="Arial") -> str

# render.py
build_ffmpeg_cmd(src, start, end, ass, out, width, height, crf, preset) -> list[str]
render_clip(cmd) -> None
```

## Design decisions

- **Embedder injection**: every function that needs vectors takes an `Embedder`, so tests use a
  fake one and never download models.
- **Determinism**: greedy decoding in Whisper, no randomness in boundary detection or selection,
  stable tie-breaking, seeded numpy. Same input + same models => same candidates.
- **Center crop** in milestone 1. Face/speaker-aware cropping replaces the crop expression in
  `render.build_ffmpeg_cmd` later.
- **Captions** are ASS so word-level highlighting (`\k` karaoke tags) can be added without
  changing the render step.
