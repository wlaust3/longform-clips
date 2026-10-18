"""Stage 7: build ASS subtitles (clip-relative timing) for burning in with FFmpeg."""

from __future__ import annotations

from dataclasses import dataclass

from .models import Word

PUNCT_BREAK = (",", ".", "?", "!", ";", ":", "，", "。", "？", "！")


@dataclass(frozen=True)
class Cue:
    start: float  # seconds, relative to clip start
    end: float
    text: str


def words_to_cues(
    words: list[Word],
    clip_start: float,
    clip_end: float,
    max_words: int = 4,
    max_chars: int = 28,
    min_duration: float = 0.3,
) -> list[Cue]:
    """Group words inside [clip_start, clip_end] into short caption lines.

    Times are shifted to be relative to the clip. A cue breaks at max_words, max_chars or
    punctuation; each cue stays visible until the next one starts (no flicker).
    """
    inside = [w for w in words if w.end > clip_start and w.start < clip_end]
    cues: list[Cue] = []
    cur: list[Word] = []

    def flush() -> None:
        if not cur:
            return
        s = max(cur[0].start, clip_start) - clip_start
        e = min(cur[-1].end, clip_end) - clip_start
        cues.append(Cue(s, max(e, s + min_duration), " ".join(w.text for w in cur)))
        cur.clear()

    for w in inside:
        text_len = sum(len(x.text) + 1 for x in cur) + len(w.text)
        if cur and (len(cur) >= max_words or text_len > max_chars):
            flush()
        cur.append(w)
        if w.text.endswith(PUNCT_BREAK):
            flush()
    flush()
    # Hold each cue until the next begins (up to 0.5 s) and never overlap.
    out: list[Cue] = []
    for i, c in enumerate(cues):
        end = c.end
        if i + 1 < len(cues):
            nxt = cues[i + 1].start
            end = min(max(end, min(nxt, c.end + 0.5)), nxt)
        out.append(Cue(c.start, end, c.text))
    return out


def ass_time(t: float) -> str:
    """H:MM:SS.cc"""
    cs = int(round(t * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", "\\N")


def build_ass(cues: list[Cue], width: int = 1080, height: int = 1920, font: str = "Arial") -> str:
    """Return an ASS document. Sizes scale with the output height (72px at 1920)."""
    fs = round(height * 72 / 1920)
    outline = max(2, round(height * 5 / 1920))
    margin_v = round(height * 360 / 1920)  # keep clear of platform UI at the bottom
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font},{fs},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,"
        f"100,100,0,0,1,{outline},1,2,60,60,{margin_v},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )
    lines = [
        f"Dialogue: 0,{ass_time(c.start)},{ass_time(c.end)},Default,,0,0,0,,{_escape(c.text)}"
        for c in cues
    ]
    return header + "\n".join(lines) + ("\n" if lines else "")
