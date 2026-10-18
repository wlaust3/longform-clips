"""Stage 4: text embeddings. Real model is lazy-imported; a hashing embedder is built in."""

from __future__ import annotations

import re
import zlib
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .models import LongformClipsError

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, d) float array of L2-normalised row vectors."""
        ...


class HashingEmbedder:
    """Dependency-free bag-of-words hashing embedder.

    Deterministic and offline: used in tests and as a fallback (`--embedder hash`).
    Much weaker than a real sentence model.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[\w']+", t.lower()):
                out[i, zlib.crc32(tok.encode("utf-8")) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise LongformClipsError(
                "sentence-transformers is not installed. Run: pip install -e '.[ml]' "
                "(or use --embedder hash for a weak offline fallback)."
            ) from e
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True))


def get_embedder(name: str) -> Embedder:
    if name == "hash":
        return HashingEmbedder()
    if name in ("minilm", "default"):
        return SentenceTransformerEmbedder()
    return SentenceTransformerEmbedder(name)
