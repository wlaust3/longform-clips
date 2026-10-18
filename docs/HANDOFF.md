# Handoff notes for coding agents and contributors

## What this is

Longform Clips: long video -> Whisper transcript -> embedding-based topic boundaries -> best
30-60 s moments -> vertical 9:16 clips with burned-in captions. Local CLI, Python 3.11+.
Status: pre-alpha. The skeleton, pure-logic stages (chunk, boundaries, select, captions) and
FFmpeg render already exist with tests. What is **not yet verified** is the real-model path
(faster-whisper and sentence-transformers on real audio) and quality on real videos.

Read first: README.md, docs/ARCHITECTURE.md, docs/MILESTONE-1.md.

## In scope (milestone 1)

Everything in docs/MILESTONE-1.md. Tracked as issues under the "Milestone 1" GitHub milestone.

## Explicitly out of scope

- Deployment, hosting, GitHub Pages, Docker publishing, servers
- Accounts, authentication, billing
- Social posting or scheduling APIs (TikTok, YouTube, Instagram, etc.)
- Face tracking / speaker-aware cropping (later milestone; center crop only now)
- LLM-based hook/virality scoring (later milestone)
- Languages beyond what Whisper does by default (no custom translation or language UI)
- Word-level caption highlighting (later milestone; keep ASS generation extensible)

Do not add any of the above, even as "small extras". Open an issue instead.

## Conventions

- Python 3.11+, type hints, `from __future__ import annotations`.
- Package `longform_clips` under `src/`; CLI command `longform-clips` (subcommands `analyze`, `preview`, `export`).
- Lint and format with ruff (`make lint`, `make fmt`); line length 100.
- Heavy deps (`faster-whisper`, `sentence-transformers`) are imported lazily and live in the
  `[ml]` extra. Never import them at module top level. CI must not download models.
- User-facing failures raise `LongformClipsError`; the CLI prints `error: ...` and exits 1. No tracebacks for expected errors.
- Anything needing vectors takes an `Embedder`; tests use the fake embedder in `tests/conftest.py`.
- Keep outputs deterministic (greedy decoding, stable sorting, seeded RNG).
- JSON files carry `schema_version`. Changing a format means updating docs/ARCHITECTURE.md and bumping the version.
- Only use test media you generate (ffmpeg `lavfi`) or own. Do not commit videos.

## Suggested build order

1. Ingest + `transcribe` against a real short video; confirm word timestamps look right.
2. Verify `chunk` / `embed` / `boundaries` on a real transcript; tune window, chunk size, cutoff.
3. Tune `select` scoring; make sure 3-5 good candidates come out of a 30-minute video.
4. Captions: check the style on real renders; verify sync within 200 ms.
5. Render + preview + export polish, error messages.
6. Performance measurement on a 30-minute video; record numbers in the README.
7. Tests and CI hardening.

## Definition of done

See "Definition of done" in docs/MILESTONE-1.md. In short: all acceptance boxes checked, CI
green on `main`, README matches behavior, and one real video verified by hand.

## Workflow

Branch from `main`, open a PR, keep CI green, reference the issue (`Refs #N`). Do not close the
milestone issue until every box is checked.
