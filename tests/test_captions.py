from longform_clips.captions import Cue, ass_time, build_ass, words_to_cues
from longform_clips.models import Word


def _words():
    return [Word(10 + i * 0.4, 10.4 + i * 0.4, f"w{i}") for i in range(10)]


def test_ass_time_format():
    assert ass_time(0) == "0:00:00.00"
    assert ass_time(65.256) == "0:01:05.26"
    assert ass_time(3600) == "1:00:00.00"


def test_cues_are_clip_relative_and_ordered():
    cues = words_to_cues(_words(), clip_start=10, clip_end=14)
    assert cues[0].start == 0
    assert all(c.end > c.start for c in cues)
    assert all(a.end <= b.start + 1e-9 for a, b in zip(cues, cues[1:], strict=False))
    assert max(len(c.text.split()) for c in cues) <= 4


def test_cues_sync_within_200ms():
    words = _words()
    cues = words_to_cues(words, 10, 14)
    # first word of each cue starts within 0.2 s of its source word
    idx = 0
    for c in cues:
        w = words[idx]
        assert abs((w.start - 10) - c.start) < 0.2
        idx += len(c.text.split())


def test_words_outside_clip_excluded():
    cues = words_to_cues(_words(), 11, 12)
    assert all("w0" not in c.text.split() for c in cues)


def test_punctuation_breaks_cue():
    ws = [Word(0, 0.3, "Hi,"), Word(0.3, 0.6, "there"), Word(0.6, 0.9, "you")]
    assert [c.text for c in words_to_cues(ws, 0, 1)] == ["Hi,", "there you"]


def test_build_ass_structure():
    ass = build_ass([Cue(0, 1.5, "Hello {x}\nworld")], 1080, 1920)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert "Dialogue: 0,0:00:00.00,0:00:01.50,Default,,0,0,0,,Hello (x)\\Nworld" in ass
    assert "[Events]" in ass
