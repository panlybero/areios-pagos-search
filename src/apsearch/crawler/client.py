"""A deliberately slow, well-behaved HTTP client for areiospagos.gr.

Politeness design
-----------------
* The origin is a single IIS/ASP box run by a public institution, and there is
  no robots.txt (it 404s). No crawl budget has been granted, so we impose a
  conservative one ourselves rather than assuming permission.
* Strictly serialised: one request in flight at a time, with a mandatory sleep
  between requests (``crawl_delay``, default 2s => 0.5 req/s), plus jitter.
* Exponential backoff on 5xx/429/timeouts, honouring ``Retry-After``.
* Every response is archived, so re-runs, parser fixes and tests never re-hit
  the origin. This is the measure that matters most.
* ``max_requests_per_run`` is a hard runaway-job circuit breaker.

Encoding: pages are windows-1253 and the charset header is often missing, so we
decode explicitly instead of trusting httpx's sniffing.
"""

from __future__ import annotations

import hashlib
import random
import threading
import time
from urllib.parse import quote

import httpx

from apsearch.config import settings
from apsearch.logging import get_logger
from apsearch.storage import BlobStore, get_blob_store

log = get_logger(__name__)

SITE_ENCODING = "cp1253"
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


class CrawlBudgetExceeded(RuntimeError):
    """Raised when a run exceeds ``max_requests_per_run``."""


class RateLimiter:
    """Global minimum-interval limiter shared by every request."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_at:
                time.sleep(self._next_at - now)
            # Jitter so we never look like a metronome.
            self._next_at = time.monotonic() + self.min_interval * random.uniform(0.9, 1.3)

    def penalise(self, seconds: float) -> None:
        """Push the next allowed request further out (after a server error)."""
        with self._lock:
            self._next_at = max(self._next_at, time.monotonic() + seconds)


class PoliteClient:
    def __init__(
        self,
        base_url: str | None = None,
        delay: float | None = None,
        store: BlobStore | None = None,
        use_cache: bool = True,
    ):
        self.base_url = (base_url or settings.base_url).rstrip("/")
        self.limiter = RateLimiter(delay if delay is not None else settings.crawl_delay)
        self.use_cache = use_cache
        self.store = store if store is not None else get_blob_store()

        ua = settings.user_agent
        if settings.contact_email:
            ua = f"{ua} contact:{settings.contact_email}"
        self._client = httpx.Client(
            headers={
                "User-Agent": ua,
                "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
                "Accept-Language": "el,en;q=0.8",
            },
            timeout=settings.request_timeout,
            limits=httpx.Limits(max_connections=12, max_keepalive_connections=10),
            follow_redirects=True,
        )
        self.n_requests = 0
        self.n_cache_hits = 0

    # ------------------------------------------------------------------ cache
    @staticmethod
    def cache_key(method: str, url: str, payload: dict | None) -> str:
        raw = f"{method}\n{url}\n{sorted((payload or {}).items())}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ---------------------------------------------------------------- fetching
    def get(self, path: str, params: dict | None = None, **kw) -> str:
        return self._request("GET", path, params=params, **kw)

    def post(self, path: str, data: dict | None = None, **kw) -> str:
        return self._request("POST", path, data=data, **kw)

    def get_json(self, path: str, params: dict | None = None, **kw) -> dict:
        import json

        text = self._request("GET", path, params=params, encoding="utf-8", **kw)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            log.warning("non-JSON response from %s", path)
            return {}

    def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        data: dict | None = None,
        force_refresh: bool = False,
        encoding: str | None = SITE_ENCODING,
    ) -> str:
        url = path if path.startswith("http") else f"{self.base_url}/{path.lstrip('/')}"
        payload = dict(data or {})
        if params:
            payload.update({f"__q_{k}": v for k, v in params.items()})
        key = self.cache_key(method, url, payload)

        if self.use_cache and not force_refresh:
            cached = self.store.get(key)
            if cached is not None:
                self.n_cache_hits += 1
                return cached

        if settings.max_requests_per_run and self.n_requests >= settings.max_requests_per_run:
            raise CrawlBudgetExceeded(
                f"request budget of {settings.max_requests_per_run} exhausted"
            )

        last_exc: Exception | None = None
        for attempt in range(settings.max_retries):
            self.limiter.wait()
            try:
                self.n_requests += 1
                if method == "POST":
                    resp = self._client.post(
                        url,
                        params=params,
                        content=encode_form(data or {}),
                        headers={
                            "Content-Type": (
                                f"application/x-www-form-urlencoded; charset={SITE_ENCODING}"
                            )
                        },
                    )
                else:
                    resp = self._client.get(url, params=params)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                self._backoff(attempt, reason=repr(exc))
                continue

            if resp.status_code in RETRY_STATUS:
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code} for {url}", request=resp.request, response=resp
                )
                self._backoff(
                    attempt,
                    reason=f"HTTP {resp.status_code}",
                    retry_after=resp.headers.get("Retry-After"),
                )
                continue

            resp.raise_for_status()
            text = decode_response(resp, encoding)
            if self.use_cache:
                self.store.put(key, text)
            return text

        raise RuntimeError(
            f"giving up on {url} after {settings.max_retries} attempts"
        ) from last_exc

    def _backoff(self, attempt: int, reason: str, retry_after: str | None = None) -> None:
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = 60.0
        else:
            # 5s, 15s, 45s, 135s -- back right off; the origin is not ours.
            delay = 5.0 * (3**attempt) * random.uniform(0.8, 1.2)
        log.warning("backing off %.1fs (attempt %d/%d): %s",
                    delay, attempt + 1, settings.max_retries, reason)
        # Also slow down every *other* caller, not just this one.
        self.limiter.penalise(delay)
        time.sleep(delay)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> PoliteClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def decode_response(resp: httpx.Response, encoding: str | None) -> str:
    if encoding is None:
        return resp.text
    ctype = resp.headers.get("Content-Type", "").lower()
    if "json" in ctype or "utf-8" in ctype:
        return resp.content.decode("utf-8", errors="replace")
    return resp.content.decode(encoding, errors="replace")


def encode_form(data: dict) -> bytes:
    """URL-encode a form body using the site's legacy codepage."""
    parts = []
    for k, v in data.items():
        ks = quote(str(k).encode(SITE_ENCODING, errors="replace"))
        vs = quote(str(v).encode(SITE_ENCODING, errors="replace"))
        parts.append(f"{ks}={vs}")
    return "&".join(parts).encode("ascii")
