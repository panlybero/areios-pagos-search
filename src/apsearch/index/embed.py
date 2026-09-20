"""Embedding backends.

Default is ``intfloat/multilingual-e5-small`` served through fastembed/ONNX:

* genuinely trained for a 512-token window -- this matters, because the popular
  ``paraphrase-multilingual-MiniLM-L12-v2`` truncates at 128 tokens, which
  silently discards most of a ~1400-character Greek legal chunk;
* 384 dimensions keeps the vector index small and similarity lookups fast;
* ONNX on CPU means no torch dependency and no GPU requirement.

Measured on a 4-vCPU box with no AVX-512: ~3.8 chunks/s fp32, ~4.9 int8. On a
GCP N2/C3 instance (AVX-512 VNNI) the int8 model is several times faster, which
is why the ONNX file is configurable rather than hard-coded.

e5 is an *asymmetric* model: passages must be prefixed "passage: " and queries
"query: ". Getting this wrong quietly degrades recall, so the prefixes live in
config next to the model name and are applied here, in one place.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterable, Sequence
from typing import Protocol

from apsearch.config import settings
from apsearch.logging import get_logger

log = get_logger(__name__)


class EmbeddingBackend(Protocol):
    name: str
    dim: int

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

#: Models fastembed does not ship in its catalogue but that we want available.
#: dim / pooling / normalisation are properties of the checkpoint, not choices.
CUSTOM_MODELS: dict[str, dict] = {
    "intfloat/multilingual-e5-small": {
        "hf": "intfloat/multilingual-e5-small",
        "dim": 384,
        "model_file": "onnx/model.onnx",
    },
    "intfloat/multilingual-e5-small-int8": {
        "hf": "intfloat/multilingual-e5-small",
        "dim": 384,
        "model_file": "onnx/model_qint8_avx512_vnni.onnx",
    },
    "intfloat/multilingual-e5-base": {
        "hf": "intfloat/multilingual-e5-base",
        "dim": 768,
        "model_file": "onnx/model.onnx",
    },
    "intfloat/multilingual-e5-base-int8": {
        "hf": "intfloat/multilingual-e5-base",
        "dim": 768,
        "model_file": "onnx/model_qint8_avx512_vnni.onnx",
    },
}

_registered: set[str] = set()
_register_lock = threading.Lock()


def _register_custom(name: str) -> None:
    spec = CUSTOM_MODELS.get(name)
    if spec is None:
        return
    with _register_lock:
        if name in _registered:
            return
        from fastembed import TextEmbedding
        from fastembed.common.model_description import ModelSource, PoolingType

        already = {m["model"] for m in TextEmbedding.list_supported_models()}
        if name not in already:
            TextEmbedding.add_custom_model(
                model=name,
                pooling=PoolingType.MEAN,
                normalization=True,
                sources=ModelSource(hf=spec["hf"]),
                dim=spec["dim"],
                model_file=spec["model_file"],
            )
        _registered.add(name)


class FastEmbedBackend:
    """ONNX-backed embeddings via fastembed."""

    def __init__(
        self,
        model: str | None = None,
        dim: int | None = None,
        query_prefix: str | None = None,
        passage_prefix: str | None = None,
        threads: int | None = None,
        cache_dir: str | None = None,
    ):
        from fastembed import TextEmbedding

        self.name = model or settings.embed_model
        self.dim = dim or settings.embed_dim
        self.query_prefix = (
            query_prefix if query_prefix is not None else settings.embed_query_prefix
        )
        self.passage_prefix = (
            passage_prefix if passage_prefix is not None else settings.embed_passage_prefix
        )
        _register_custom(self.name)

        kwargs = {"cache_dir": cache_dir or settings.model_cache_dir}
        n_threads = threads if threads is not None else settings.embed_threads
        if n_threads:
            kwargs["threads"] = n_threads
        self._model = TextEmbedding(self.name, **kwargs)
        log.info("embedding backend: %s (dim=%d)", self.name, self.dim)

    def _embed(self, texts: Sequence[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []
        prepared = [f"{prefix}{t}" if prefix else t for t in texts]
        vectors = list(
            self._model.embed(prepared, batch_size=settings.embed_batch_size)
        )
        out = [v.tolist() for v in vectors]
        if out and len(out[0]) != self.dim:
            raise ValueError(
                f"model {self.name} produced dim {len(out[0])}, config says {self.dim}"
            )
        return out

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, self.passage_prefix)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, self.query_prefix)


class HashingBackend:
    """Deterministic, dependency-free stand-in used by tests.

    Produces stable pseudo-vectors so the search plumbing can be exercised
    without downloading a model. Never use for real retrieval.
    """

    def __init__(self, dim: int | None = None):
        self.name = "hashing-test"
        self.dim = dim or settings.embed_dim

    def _one(self, text: str) -> list[float]:
        import math

        vec = [0.0] * self.dim
        for token in text.lower().split():
            h = int.from_bytes(hashlib.md5(token.encode()).digest()[:8], "big")
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


_backend: EmbeddingBackend | None = None
_backend_lock = threading.Lock()


def get_backend(force: EmbeddingBackend | None = None) -> EmbeddingBackend:
    """Process-wide singleton; loading an ONNX session is expensive."""
    global _backend
    if force is not None:
        _backend = force
        return _backend
    with _backend_lock:
        if _backend is None:
            _backend = _build_backend()
    return _backend


def _build_backend() -> EmbeddingBackend:
    kind = settings.embed_backend.lower()
    if kind in ("hashing-test", "test"):
        return HashingBackend()
    if kind == "gemini":
        from apsearch.index.gemini import GeminiBackend

        return GeminiBackend()
    if kind in ("fastembed", "local", "onnx"):
        return FastEmbedBackend()
    raise ValueError(
        f"unknown embed backend {settings.embed_backend!r} "
        "(expected: gemini | fastembed | hashing-test)"
    )


def reset_backend() -> None:
    global _backend
    _backend = None


def batched(items: Iterable, size: int):
    buf = []
    for item in items:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf
