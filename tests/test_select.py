from longform_clips.boundaries import detect_boundaries
from longform_clips.chunk import make_chunks, split_sentences
from longform_clips.models import Boundary
from longform_clips.pipeline import find_candidates
from longform_clips.select import SelectConfig, generate_windows, select_candidates


def test_candidates_are_30_to_60s_sentence_aligned(transcript, fake_embedder):
    cands = find_candidates(transcript, fake_embedder)
    sents = split_sentences(transcript)
    starts = {s.start for s in sents}
    ends = {s.end for s in sents}
    assert 3 <= len(cands) <= 5
    for c in cands:
        assert 30 <= c.duration <= 60
        assert c.start in starts and c.end in ends
        assert c.reason and c.text


def test_candidates_do_not_overlap(transcript, fake_embedder):
    cands = sorted(find_candidates(transcript, fake_embedder), key=lambda c: c.start)
    for a, b in zip(cands, cands[1:], strict=False):
        assert a.end <= b.start


def test_deterministic(transcript, fake_embedder):
    a = find_candidates(transcript, fake_embedder, seed=1)
    b = find_candidates(transcript, fake_embedder, seed=1)
    assert a == b


def test_boundary_alignment_raises_score(transcript):
    sents = split_sentences(transcript)
    cfg = SelectConfig(num_candidates=1)
    target = sents[10].start
    with_b = select_candidates(sents, [Boundary(target, 0, 1.0)], cfg)[0]
    assert abs(with_b.start - target) < 3 or abs(with_b.end - target) < 3


def test_short_video_gives_no_candidates():
    from tests.conftest import build_transcript

    tr = build_transcript(sentences_per_topic=2)
    sents = split_sentences(tr)
    assert generate_windows(sents, SelectConfig()) == []
    assert select_candidates(sents, []) == []


def test_boundaries_used_end_to_end(transcript, fake_embedder):
    chunks = make_chunks(split_sentences(transcript))
    assert detect_boundaries(chunks, fake_embedder.embed([c.text for c in chunks]))
