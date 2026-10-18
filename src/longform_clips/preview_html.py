"""Generate the static preview.html (relative links, no server needed)."""

from __future__ import annotations

from html import escape

from .models import Candidate

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Longform Clips preview</title>
<style>
body{{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:1rem}}
.grid{{display:flex;flex-wrap:wrap;gap:1rem}}
.card{{background:#1c1c1c;border-radius:8px;padding:.75rem;width:240px}}
video{{width:100%;aspect-ratio:9/16;background:#000;border-radius:4px}}
code{{background:#333;padding:2px 4px;border-radius:3px}} small{{color:#aaa}}
</style></head><body>
<h1>Longform Clips: candidates</h1>
<p>Pick one, then run <code>longform-clips export --out {out} --candidate N</code>.</p>
<div class="grid">
{cards}
</div></body></html>
"""

CARD = """<div class="card"><h3>Candidate {id}</h3>
<video controls preload="metadata" src="{src}"></video>
<p>{start:.1f}s &ndash; {end:.1f}s ({dur:.0f}s) &middot; score {score:.2f}</p>
<small>{reason}</small></div>"""


def build_preview_html(cands: list[Candidate], out_dir_name: str = "out") -> str:
    cards = "\n".join(
        CARD.format(
            id=c.id,
            src=f"previews/candidate_{c.id}.mp4",
            start=c.start,
            end=c.end,
            dur=c.duration,
            score=c.score,
            reason=escape(c.reason),
        )
        for c in cands
    )
    return TEMPLATE.format(cards=cards, out=escape(out_dir_name))
