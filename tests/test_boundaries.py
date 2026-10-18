import numpy as np

from longform_clips.boundaries import adjacent_similarity, depth_scores, detect_boundaries
from longform_clips.chunk import make_chunks, split_sentences


def test_depth_scores_peak_at_valley():
    sims = np.array([0.9, 0.9, 0.2, 0.9, 0.9])
    d = depth_scores(sims)
    assert d.argmax() == 2 and d[2] > 1.0


def test_adjacent_similarity_shape():
    v = np.eye(4)
    assert adjacent_similarity(v, window=1).shape == (3,)
    assert adjacent_similarity(np.eye(1)).shape == (0,)


def test_boundaries_found_near_topic_changes(transcript, fake_embedder):
    sents = split_sentences(transcript)
    chunks = make_chunks(sents)
    vecs = fake_embedder.embed([c.text for c in chunks])
    found = detect_boundaries(chunks, vecs)
    per_topic = transcript.duration / 3
    assert len(found) >= 2
    for target in (per_topic, 2 * per_topic):
        assert min(abs(b.time - target) for b in found) < 15
    assert all(0 < b.depth <= 1 for b in found)


def test_no_boundaries_for_tiny_input():
    assert detect_boundaries([], np.zeros((0, 3))) == []
