"""Gemini embedding backend.

Why this exists alongside the local ONNX backend
------------------------------------------------
Embedding ~1.3M Greek legal chunks on a 4-vCPU box takes ~95 hours with
multilingual-e5-small. The same work through Gemini's batch endpoint runs at
~70 embeddings/s per connection, i.e. a few hours -- and the quality on Greek
is substantially better.

The trade-off is a hosted dependency, so the interface is identical to the
local backend and the choice is a single config value. Switching models
invalidates every stored vector (different space, possibly different
dimension), which ``apsearch index reset`` handles explicitly.

Two details that are easy to get wrong
--------------------------------------
1. **Task types are asymmetric.** Passages must be embedded with
   ``RETRIEVAL_DOCUMENT`` and queries with ``RETRIEVAL_QUERY``. Using one type
   for both measurably degrades recall.
2. **MRL truncation denormalises.** ``gemini-embedding-001`` natively produces
   3072 dims; asking for 768 truncates and the result is *not* unit-length
   (observed norm ~0.59). Cosine distance in pgvector still works, but inner
   product and any averaging do not. We renormalise unconditionally.
"""

from __future__ import annotations

import math
import os
import random
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import httpx

from apsearch.config import settings
from apsearch.logging import get_logger

log = get_logger(__name__)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"

#: Hard cap enforced by batchEmbedContents.
MAX_BATCH = 100

#: Input token ceilings, used to pre-truncate rather than eat a 400.
MODEL_TOKEN_LIMIT = {
    "gemini-embedding-001": 2048,
    "gemini-embedding-2": 8192,
    "gemini-embedding-2-preview": 8192,
}

TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"


def _api_key() -> str:
    key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "No Gemini API key. Set APSEARCH_GEMINI_API_KEY or GEMINI_API_KEY "
            "(never commit it; use .env locally and Secret Manager on GCP)."
        )
    return key


def _normalise(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0:
        return values
    return [v / norm for v in values]


class _RateLimiter:
    """Token-bucket-ish limiter: at most `rpm` requests per minute."""

    def __init__(self, rpm: int):
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> None:
        if not self.min_interval:
            return
        with self._lock:
            now = time.monotonic()
            if now < self._next_at:
                time.sleep(self._next_at - now)
                now = time.monotonic()
            self._next_at = now + self.min_interval

    def penalise(self, seconds: float) -> None:
        with self._lock:
            self._next_at = max(self._next_at, time.monotonic() + seconds)


class GeminiBackend:
    def __init__(
        self,
        model: str | None = None,
        dim: int | None = None,
        api_key: str | None = None,
        concurrency: int | None = None,
        rpm: int | None = None,
    ):
        self.name = model or settings.gemini_embed_model
        self.dim = dim or settings.embed_dim
        self._key = api_key or _api_key()
        self.concurrency = concurrency if concurrency is not None else settings.gemini_concurrency
        self.token_limit = MODEL_TOKEN_LIMIT.get(self.name, 2048)
        #: ~2 chars/token for Greek; stay well inside the limit.
        self.char_limit = int(self.token_limit * 1.8)
        self._limiter = _RateLimiter(rpm if rpm is not None else settings.gemini_rpm)
        self._client = httpx.Client(
            timeout=httpx.Timeout(settings.gemini_timeout),
            limits=httpx.Limits(max_connections=max(self.concurrency, 1) + 2),
        )
        log.info(
            "embedding backend: gemini %s (dim=%d, concurrency=%d)",
            self.name, self.dim, self.concurrency,
        )

    # ------------------------------------------------------------------ HTTP
    def _post_batch(self, texts: Sequence[str], task: str) -> list[list[float]]:
        requests = [
            {
                "model": f"models/{self.name}",
                "content": {"parts": [{"text": t[: self.char_limit]}]},
                "taskType": task,
                "outputDimensionality": self.dim,
            }
            for t in texts
        ]
        url = f"{API_ROOT}/models/{self.name}:batchEmbedContents"

        last: Exception | None = None
        for attempt in range(settings.gemini_max_retries):
            self._limiter.wait()
            try:
                resp = self._client.post(
                    url, params={"key": self._key}, json={"requests": requests}
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = exc
                self._backoff(attempt, repr(exc))
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After")
                last = RuntimeError(f"HTTP {resp.status_code}")
                self._backoff(attempt, f"HTTP {resp.status_code}", retry_after)
                continue

            if resp.status_code >= 400:
                # 4xx other than 429 will not fix themselves.
                raise RuntimeError(
                    f"Gemini embed failed {resp.status_code}: {resp.text[:400]}"
                )

            payload = resp.json()
            embeddings = payload.get("embeddings") or []
            if len(embeddings) != len(texts):
                raise RuntimeError(
                    f"expected {len(texts)} embeddings, got {len(embeddings)}"
                )
            return [_normalise(e["values"]) for e in embeddings]

        raise RuntimeError(
            f"Gemini embed gave up after {settings.gemini_max_retries} attempts"
        ) from last

    def _backoff(self, attempt: int, reason: str, retry_after: str | None = None) -> None:
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = 30.0
        else:
            delay = min(60.0, 2.0 * (2**attempt)) * random.uniform(0.8, 1.2)
        log.warning("gemini backoff %.1fs (attempt %d): %s", delay, attempt + 1, reason)
        self._limiter.penalise(delay)
        time.sleep(delay)

    # ------------------------------------------------------------- interface
    def _embed(self, texts: Sequence[str], task: str) -> list[list[float]]:
        if not texts:
            return []
        batches = [
            list(texts[i : i + MAX_BATCH]) for i in range(0, len(texts), MAX_BATCH)
        ]
        if len(batches) == 1 or self.concurrency <= 1:
            out: list[list[float]] = []
            for b in batches:
                out.extend(self._post_batch(b, task))
            return out

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            results = list(pool.map(lambda b: self._post_batch(b, task), batches))
        return [vec for batch in results for vec in batch]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, TASK_DOCUMENT)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, TASK_QUERY)

    def close(self) -> None:
        self._client.close()
