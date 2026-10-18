from longform_clips.chunk import make_chunks, split_sentences
from longform_clips.models import Segment, Transcript, Word


def test_split_sentences_on_punctuation_and_gap():
    words = [Word(0, 0.5, "Hello"), Word(0.5, 1, "world."), Word(1.1, 1.5, "Next"),
             Word(4.0, 4.4, "after"), Word(4.4, 4.8, "gap")]  # fmt: skip
    tr = Transcript("en", 5, [Segment(0, 4.8, "x", words)])
    s = split_sentences(tr, max_gap=1.0)
    assert [x.text for x in s] == ["Hello world.", "Next", "after gap"]
    assert s[0].start == 0 and s[0].end == 1


def test_sentences_cap_length():
    words = [Word(i * 0.5, i * 0.5 + 0.5, f"w{i}") for i in range(100)]
    tr = Transcript("en", 50, [Segment(0, 50, "x", words)])
    s = split_sentences(tr, max_sentence_seconds=10)
    assert len(s) >= 5 and all(x.end - x.start <= 10.5 for x in s)


def test_make_chunks_cover_all_sentences_in_order(transcript):
    sents = split_sentences(transcript)
    chunks = make_chunks(sents, target_seconds=12)
    assert chunks[0].sentence_start == 0 and chunks[-1].sentence_end == len(sents)
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a.sentence_end == b.sentence_start and a.end <= b.start
    assert [c.index for c in chunks] == list(range(len(chunks)))
