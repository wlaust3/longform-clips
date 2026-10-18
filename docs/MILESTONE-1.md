# Milestone 1: One video in, 3-5 candidate clips out, export one

## Goal

A working CLI that takes one long video (local file or URL), finds 3-5 candidate 30-60 s
moments, lets you preview them as vertical clips in a browser, and exports one chosen clip as
a final 1080x1920 mp4 with burned-in captions.

```bash
longform-clips analyze <video|url> --out out/   # transcript.json + candidates.json
longform-clips preview --out out/               # previews/ + preview.html
longform-clips export --out out/ --candidate N  # clips/clip_N.mp4
```

## Scope

- Python 3.11+ package `longform_clips`, CLI `longform-clips` (Typer) with `analyze`, `preview`, `export`
- Input: local file or URL via yt-dlp
- STT: faster-whisper with word timestamps
- Sentence/chunk splitting, embeddings (sentence-transformers MiniLM), TextTiling-style boundary detection
- Heuristic candidate scoring and selection (see docs/ARCHITECTURE.md)
- ASS captions, FFmpeg center-crop 9:16 render, burned-in captions
- Generated static `preview.html`
- Unit tests, one end-to-end smoke test, CI

## Out of scope

Deployment or hosting, web UI server, accounts, billing, social posting/scheduling APIs,
face/speaker tracking (center crop only), LLM scoring, word-level highlight animation, and
languages beyond what Whisper handles by default.

## Acceptance criteria

### Analyze
- [ ] `longform-clips analyze <video|url> --out out/` writes `out/transcript.json` and `out/candidates.json` matching docs/ARCHITECTURE.md
- [ ] `candidates.json` has 3-5 candidates for a video of at least 5 minutes of speech, each 30-60 s long
- [ ] Every candidate starts at a sentence start and ends at a sentence end, and candidates do not overlap
- [ ] Each candidate has a `score` and a human-readable `reason`
- [ ] Running `analyze` twice on the same input with the same `--seed` and models gives identical `candidates.json`
- [ ] URL input downloads via yt-dlp into `out/source/`

### Preview
- [ ] `longform-clips preview --out out/` writes `out/preview.html` with one playable `<video>` per candidate (relative links; opens by double-click, no server)
- [ ] Preview clips are 9:16 (low resolution is fine, e.g. 360x640) with captions burned in

### Export
- [ ] `longform-clips export --out out/ --candidate N` writes `out/clips/clip_N.mp4`, H.264/AAC, exactly 1080x1920, with burned-in captions
- [ ] An unknown candidate id fails with a message listing the available ids

### Caption quality
- [ ] Captions are readable (large, white with outline, kept clear of the bottom UI area) and never more than 2 lines
- [ ] Caption cues start within 200 ms of the first spoken word of the cue (verified by a unit test on cue timing and a manual check on a real clip)

### Robustness
- [ ] Missing ffmpeg/ffprobe gives a clear error with install hints, exit code 1, no traceback
- [ ] Bad URL, unreachable URL, or a non-existent file gives a clear error, exit code 1, no traceback
- [ ] A video too short for 3 candidates prints a warning instead of crashing

### Performance
- [ ] Documented in the README: on a CPU-only machine, `analyze` of a 30-minute video with `--model small` finishes within a stated time budget (target: under 45 minutes on a 4-core laptop; measured number recorded in the README), and `preview` + `export` take under 5 minutes

### Tests and CI
- [ ] Unit tests for chunking, boundary detection, candidate selection, and caption/ASS generation use fixtures only and inject a fake embedder (no model downloads)
- [ ] One end-to-end smoke test on a short generated sample video (ffmpeg `lavfi`) covers analyze -> preview -> export using a stored transcript
- [ ] CI (GitHub Actions) installs the package, runs `ruff check`, `ruff format --check` and `pytest` on Python 3.11-3.13, and is green on `main`
- [ ] `pip install -e ".[dev]"` and `longform-clips --help` work in a fresh venv

## Definition of done

All boxes above are checked, CI is green on `main`, README usage matches actual behavior, and
at least one real podcast or interview video has been run end to end by hand with the result
noted in the PR description (real Whisper + MiniLM, not fakes).
