"""Stage 1: resolve input (local file or URL via yt-dlp) and check prerequisites."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from .models import LongformClipsError

URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def is_url(s: str) -> bool:
    return bool(URL_RE.match(s))


def require_ffmpeg() -> None:
    """Raise a clear error if ffmpeg/ffprobe are missing."""
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        raise LongformClipsError(
            f"Missing required tool(s): {', '.join(missing)}. Install FFmpeg "
            "(macOS: `brew install ffmpeg`; Debian/Ubuntu: `sudo apt install ffmpeg`; "
            "Windows: `winget install ffmpeg`) and make sure it is on your PATH."
        )


def probe_duration(path: Path) -> float:
    require_ffmpeg()
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise LongformClipsError(f"ffprobe could not read {path}: {proc.stderr.strip()}")
    return float(json.loads(proc.stdout)["format"]["duration"])


def download(url: str, dest_dir: Path) -> Path:
    """Download `url` with yt-dlp into `dest_dir` and return the resulting mp4 path."""
    import yt_dlp
    from yt_dlp.utils import DownloadError

    dest_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(dest_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
    except DownloadError as e:
        raise LongformClipsError(f"Could not download {url}: {e}") from e
    if not path.exists():
        raise LongformClipsError(f"yt-dlp finished but no file was found at {path}")
    return path


def resolve_input(src: str, out_dir: Path) -> Path:
    """Return a local video path for a file path or a URL."""
    require_ffmpeg()
    if is_url(src):
        return download(src, out_dir / "source")
    p = Path(src).expanduser()
    if not p.is_file():
        raise LongformClipsError(f"Input is not a file or an http(s) URL: {src}")
    return p
