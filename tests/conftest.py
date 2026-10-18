"""Shared fixtures: a synthetic two-topic transcript (no models, no media)."""

from __future__ import annotations

import numpy as np
import pytest

from longform_clips.models import Segment, Transcript, Word

TOPIC_A = "cats purr and nap in sunny windows while kittens chase yarn."
TOPIC_B = "interest rates and bond yields move with central bank policy."
TOPIC_C = "marathon runners train mileage and pace for race day."


def build_transcript(topics=(TOPIC_A, TOPIC_B, TOPIC_C), sentences_per_topic=12, sec_per_word=0.4):
    """Each sentence is the topic text; words are evenly spaced; 0.5 s pause between sentences."""
    t = 0.0
    segments = []
    for topic in topics:
        for _ in range(sentences_per_topic):
            words = []
            start = t
            for tok in topic.split():
                words.append(Word(round(t, 3), round(t + sec_per_word, 3), tok))
                t += sec_per_word
            segments.append(Segment(start, t, topic, words))
            t += 0.5
    return Transcript(language="en", duration=t, segments=segments)


class TopicEmbedder:
    """Fake embedder: one-hot vector per topic keyword. Needs no model download."""

    KEYS = ("cats", "rates", "marathon")

    def embed(self, texts):
        out = np.zeros((len(texts), len(self.KEYS)))
        for i, text in enumerate(texts):
            for k, key in enumerate(self.KEYS):
                out[i, k] = text.count(key)
        n = np.linalg.norm(out, axis=1, keepdims=True)
        n[n == 0] = 1
        return out / n


@pytest.fixture
def transcript():
    return build_transcript()


@pytest.fixture
def fake_embedder():
    return TopicEmbedder()
