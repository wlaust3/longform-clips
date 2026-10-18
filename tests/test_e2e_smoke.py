"""End-to-end smoke test on a generated 75 s sample video (needs ffmpeg with libass)."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest
from typer.testing import CliRunner

from longform_clips import models
from longform_clips.cli import app
from tests.conftest import build_transcript

pytestmark = pytest.mark.e2e
runner = CliRunner()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_analyze_preview_export(tmp_path):
    video = tmp_path / "sample.mp4"
    tr = build_transcript(sentences_per_topic=6)  # ~ 75 s
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         f"testsrc=size=640x360:rate=15:duration={int(tr.duration) + 1}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={int(tr.duration) + 1}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(video)],
        check=True,
    )  # fmt: skip
    tfile = tmp_path / "t.json"
    models.write_json(tfile, tr.to_dict())
    out = tmp_path / "out"

    r = runner.invoke(
        app, ["analyze", str(video), "--out", str(out), "--embedder", "hash",
              "--transcript", str(tfile), "--num", "3"],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    cands = json.loads((out / "candidates.json").read_text())["candidates"]
    assert 1 <= len(cands) <= 3
    assert all(30 <= c["end"] - c["start"] <= 60 for c in cands)

    r = runner.invoke(app, ["preview", "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert (out / "preview.html").read_text().count("<video") == len(cands)
    assert all((out / "previews" / f"candidate_{c['id']}.mp4").stat().st_size > 0 for c in cands)

    r = runner.invoke(app, ["export", "--candidate", "1", "--out", str(out)])
    assert r.exit_code == 0, r.output
    clip = out / "clips" / "clip_1.mp4"
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(clip)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert probe.stdout.strip() == "1080,1920"
    assert "Dialogue:" in (out / "clips" / "clip_1.ass").read_text()
