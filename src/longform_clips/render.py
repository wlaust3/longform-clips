"""Stage 8: cut, crop to 9:16, scale and burn captions with FFmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .models import LongformClipsError


def _filter_escape(path: str) -> str:
    """Escape a path for use inside an ffmpeg filter argument."""
    return path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace(",", "\\,")


def build_ffmpeg_cmd(
    src: Path,
    start: float,
    end: float,
    ass: Path,
    out: Path,
    width: int = 1080,
    height: int = 1920,
    crf: int = 20,
    preset: str = "medium",
) -> list[str]:
    """Build the ffmpeg command. Milestone 1 uses a centered 9:16 crop."""
    vf = (
        "crop='min(iw,ih*9/16)':'min(ih,iw*16/9)',"
        f"scale={width}:{height},setsar=1,"
        f"subtitles='{_filter_escape(str(ass))}'"
    )
    return [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
        str(out),
    ]  # fmt: skip


def render_clip(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise LongformClipsError(f"ffmpeg failed:\n{proc.stderr.strip()}")
