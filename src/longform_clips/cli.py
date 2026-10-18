"""Command line interface: `longform-clips analyze | preview | export`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from . import __version__
from .embed import get_embedder
from .ingest import probe_duration, require_ffmpeg, resolve_input
from .models import (
    Candidate,
    LongformClipsError,
    Transcript,
    candidates_from_json,
    candidates_to_json,
    read_json,
    write_json,
)
from .pipeline import find_candidates, render_candidate
from .preview_html import build_preview_html
from .select import SelectConfig
from .transcribe import transcribe

app = typer.Typer(
    help="Longform Clips: turn a long video into vertical 9:16 clips with burned-in captions.",
    no_args_is_help=True,
    add_completion=False,
)

OutOpt = Annotated[Path, typer.Option("--out", "-o", help="Output directory.")]


def _version(value: bool) -> None:
    if value:
        typer.echo(f"longform-clips {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    """Longform Clips (pre-alpha)."""


def _fail(e: LongformClipsError) -> typer.Exit:
    typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
    return typer.Exit(code=1)


def _load(out: Path) -> tuple[Transcript, list[Candidate], Path]:
    transcript = Transcript.from_dict(read_json(out / "transcript.json"))
    cands = candidates_from_json(read_json(out / "candidates.json"))
    src = Path(read_json(out / "source.json")["path"])
    if not src.is_file():
        raise LongformClipsError(f"Source video no longer exists: {src}")
    return transcript, cands, src


@app.command()
def analyze(
    source: Annotated[str, typer.Argument(help="Local video file or http(s) URL (yt-dlp).")],
    out: OutOpt = Path("out"),
    model: Annotated[str, typer.Option(help="faster-whisper model size or path.")] = "small",
    device: Annotated[str, typer.Option(help="auto, cpu or cuda.")] = "auto",
    embedder: Annotated[
        str, typer.Option(help="'minilm' (sentence-transformers), 'hash' (offline), or a model id.")
    ] = "minilm",
    num: Annotated[int, typer.Option(help="Max candidates (3-5 recommended).")] = 5,
    seed: Annotated[int, typer.Option(help="Seed for determinism.")] = 0,
    transcript_file: Annotated[
        Path | None, typer.Option("--transcript", help="Reuse an existing transcript.json.")
    ] = None,
) -> None:
    """Transcribe, find topic boundaries and write transcript.json + candidates.json."""
    try:
        src = resolve_input(source, out)
        if transcript_file:
            transcript = Transcript.from_dict(read_json(transcript_file))
        else:
            typer.echo("Transcribing (first run downloads the Whisper model)...")
            transcript = transcribe(src, model_size=model, device=device)
        transcript.source = str(src)
        if not transcript.duration:
            transcript.duration = probe_duration(src)
        cfg = SelectConfig(num_candidates=num)
        cands = find_candidates(transcript, get_embedder(embedder), cfg, seed)
        write_json(out / "transcript.json", transcript.to_dict())
        write_json(out / "candidates.json", candidates_to_json(cands))
        write_json(out / "source.json", {"path": str(Path(src).resolve())})
    except LongformClipsError as e:
        raise _fail(e) from e
    if len(cands) < cfg.min_candidates:
        typer.secho(
            f"warning: only {len(cands)} candidate(s) found (video may be too short).",
            fg=typer.colors.YELLOW,
            err=True,
        )
    for c in cands:
        typer.echo(f"#{c.id}  {c.start:7.1f}-{c.end:7.1f}s  score {c.score:.2f}  {c.text[:60]}")
    typer.echo(f"Wrote {out}/transcript.json and {out}/candidates.json")


@app.command()
def preview(out: OutOpt = Path("out")) -> None:
    """Render low-res 9:16 previews and write preview.html."""
    try:
        require_ffmpeg()
        transcript, cands, src = _load(out)
        for c in cands:
            typer.echo(f"Rendering preview {c.id}/{len(cands)}...")
            render_candidate(
                transcript, c, src, out / "previews" / f"candidate_{c.id}.mp4",
                360, 640, crf=30, preset="veryfast",
            )  # fmt: skip
        html = out / "preview.html"
        html.write_text(build_preview_html(cands, str(out)), encoding="utf-8")
    except LongformClipsError as e:
        raise _fail(e) from e
    typer.echo(f"Open {html} in a browser.")


@app.command()
def export(
    candidate: Annotated[int, typer.Option("--candidate", "-c", help="Candidate id.")],
    out: OutOpt = Path("out"),
) -> None:
    """Render the chosen candidate as a final 1080x1920 mp4 with burned-in captions."""
    try:
        require_ffmpeg()
        transcript, cands, src = _load(out)
        match = next((c for c in cands if c.id == candidate), None)
        if match is None:
            ids = ", ".join(str(c.id) for c in cands)
            raise LongformClipsError(f"No candidate {candidate}. Available: {ids}")
        dest = render_candidate(
            transcript, match, src, out / "clips" / f"clip_{candidate}.mp4", 1080, 1920
        )
    except LongformClipsError as e:
        raise _fail(e) from e
    typer.echo(f"Exported {dest}")
