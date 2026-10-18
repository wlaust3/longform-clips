from pathlib import Path

import pytest
from typer.testing import CliRunner

from longform_clips import models
from longform_clips.cli import app
from longform_clips.embed import HashingEmbedder
from longform_clips.models import Candidate, Transcript
from longform_clips.render import build_ffmpeg_cmd

runner = CliRunner()


def test_help_lists_subcommands():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    for cmd in ("analyze", "preview", "export"):
        assert cmd in r.output


def test_transcript_roundtrip(transcript):
    again = Transcript.from_dict(transcript.to_dict())
    assert again == transcript


def test_candidates_roundtrip():
    c = [Candidate(1, 1.0, 40.0, 0.5, "r", "t")]
    assert models.candidates_from_json(models.candidates_to_json(c)) == c


def test_hashing_embedder_deterministic_and_normalised():
    e = HashingEmbedder()
    a = e.embed(["hello world", "hello world"])
    assert (a[0] == a[1]).all() and abs((a[0] ** 2).sum() - 1) < 1e-9


def test_ffmpeg_cmd_targets_1080x1920():
    cmd = build_ffmpeg_cmd(Path("in.mp4"), 1, 31, Path("a.ass"), Path("o.mp4"))
    vf = cmd[cmd.index("-vf") + 1]
    assert "scale=1080:1920" in vf and "subtitles=" in vf


def test_missing_input_is_clear_error(tmp_path):
    r = runner.invoke(app, ["analyze", str(tmp_path / "nope.mp4"), "--out", str(tmp_path / "o")])
    assert r.exit_code == 1 and "not a file" in r.output


def test_missing_ffmpeg_is_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    r = runner.invoke(app, ["preview", "--out", str(tmp_path)])
    assert r.exit_code == 1 and "Install FFmpeg" in r.output


def test_export_unknown_candidate(tmp_path, transcript):
    models.write_json(tmp_path / "transcript.json", transcript.to_dict())
    models.write_json(tmp_path / "candidates.json", models.candidates_to_json([]))
    models.write_json(tmp_path / "source.json", {"path": __file__})
    if not __import__("shutil").which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    r = runner.invoke(app, ["export", "--candidate", "9", "--out", str(tmp_path)])
    assert r.exit_code == 1 and "No candidate 9" in r.output
